"""
Definition of the parent MyriadObject, from which all Myriad types inherit.
"""
import logging
import os
from pprint import pprint
from collections import OrderedDict
from pkg_resources import resource_string
from pycparser.c_ast import ArrayDecl

from .myriad_mako_wrapper import MakoTemplate, MakoFileTemplate
from .myriad_types import MyriadScalar, MyriadFunction
from .myriad_types import MVoid, MVarArgs, MInt
from .myriad_types import c_decl_to_pybuildarg
from .myriad_metaclass import _myriadclass_method, _MyriadObjectBase
from .myriad_metaclass import myriad_method_verbatim, MyriadMetaclass
from .myriad_metaclass import create_delegator, create_super_delegator
from .myriad_utils import get_all_subclasses

#######
# Log #
#######

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


#############################
# Template Resource Strings #
#############################

MYRIADOBJECT_PYC_FILE_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "pymyriadobject.c.mako")).decode("UTF-8")

MYRIADOBJECT_PYH_FILE_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "pymyriadobject.h.mako")).decode("UTF-8")

CTOR_TEMPLATE_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "ctor_template.mako")).decode("UTF-8")

INIT_OB_FUN_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "init_ob_fun.mako")).decode("UTF-8")

CUH_FILE_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "cuh_file.mako")).decode("UTF-8")

CU_FILE_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "cu_file.mako")).decode("UTF-8")

PYC_COMP_FILE_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "pyc_file.mako")).decode("UTF-8")


class MyriadObject(_MyriadObjectBase,
                   metaclass=MyriadMetaclass):
    """ Base class that every myriad object inherits from """

    @myriad_method_verbatim
    def ctor(self,
             app: MyriadScalar("app", MVarArgs, ptr=True)
             ) -> MyriadScalar('', MVoid, ptr=True):
        """    return self;"""

    @myriad_method_verbatim
    def dtor(self) -> MInt:
        """
    _my_free(self);
    return 0;
        """

    @myriad_method_verbatim
    def cudafy(self, clobber: MInt) -> MyriadScalar('', MVoid, ptr=True):
        """
    #ifdef CUDA
    struct MyriadObjectClass* _self = (struct MyriadObjectClass*) self;
    void* n_dev_obj = NULL;
    size_t my_size = myriad_size_of(self);

    const struct MyriadObjectClass* tmp = _self->m_class;
    _self->m_class = _self->m_class->device_class;

    CUDA_CHECK_RETURN(cudaMalloc(&n_dev_obj, my_size));

    CUDA_CHECK_RETURN(
        cudaMemcpy(
            n_dev_obj,
            _self,
            my_size,
            cudaMemcpyHostToDevice
            )
        );

    _self->m_class = tmp;

    return n_dev_obj;
    #else
    return NULL;
    #endif
        """

    @myriad_method_verbatim
    def decudafy(self, cuda_self: MyriadScalar("cuda_self", MVoid, ptr=True)):
        """    return;"""

    @_myriadclass_method
    def cls_ctor(self,
                 app: MyriadScalar("app", MVarArgs, ptr=True)
                 ) -> MyriadScalar('', MVoid, ptr=True):
        """
    struct MyriadObjectClass* _self = (struct MyriadObjectClass*) self;
    const size_t offset = offsetof(struct MyriadObjectClass, my_ctor_t);

    _self->super = va_arg(*app, struct MyriadObjectClass*);
    _self->size = va_arg(*app, size_t);

    assert(_self->super);

    memcpy((char*) _self + offset,
           (char*) _self->super + offset,
           myriad_size_of(_self->super) - offset);

    va_list ap;
    va_copy(ap, *app);

    voidf selector = NULL; selector = va_arg(ap, voidf);

    while (selector)
    {
        const voidf curr_method = va_arg(ap, voidf);
        if (selector == (voidf) ctor)
        {
            *(voidf *) &_self->my_ctor_t = curr_method;
        } else if (selector == (voidf) cudafy) {
            *(voidf *) &_self->my_cudafy_t = curr_method;
        } else if (selector == (voidf) dtor) {
            *(voidf *) &_self->my_dtor_t = curr_method;
        } else if (selector == (voidf) decudafy) {
            *(voidf *) &_self->my_decudafy_t = curr_method;
        }
        selector = va_arg(ap, voidf);
    }
    return _self;
        """

    @_myriadclass_method
    def cls_dtor(self) -> MInt:
        """
    fprintf(stderr, "Destroying a Class is undefined behavior.");
    return -1;
        """

    @_myriadclass_method
    def cls_cudafy(self, clobber: MInt) -> MyriadScalar('', MVoid, ptr=True):
        """
    /*
     * Invariants/Expectations:
     *
     * A) The class we're given (_self) is fully initialized on the CPU
     * B) _self->device_class == NULL, will receive this fxn's result
     * C) _self->super has been set with (void*) SuperClass->device_class
     *
     * The problem here is that we're currently ignoring anything the
     * extended class passes up at us through super_, and so we're only
     * copying the c_class struct, not the rest of the class. To solve this,
     * what we need to do is to:
     *
     * 1) Memcopy the ENTIRETY of the old class onto a new heap pointer
     *     - This works because the extended class has already made any
     *       and all of their pointers/functions CUDA-compatible.
     * 2) Alter the "top-part" of the copied-class to go to CUDA
     *     - cudaMalloc the future location of the class on the device
     *     - Set our internal object's class pointer to that location
     * 3) Copy our copied-class to the device
     * 3a) Free our copied-class
     * 4) Return the device pointer to whoever called us
     *
     * Q: How do we keep track of on-device super class?
     * A: We take it on good faith that the under class has set their supercls
     *    to be the visible SuperClass->device_class.
     */
    #ifdef CUDA
    struct MyriadObjectClass* _self = (struct MyriadObjectClass*) self;

    const struct MyriadObjectClass* dev_class = NULL;

    // DO NOT USE sizeof(struct MyriadObjectClass)!
    const size_t class_size = myriad_size_of(_self);

    // Allocate space for new class on the card
    CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_class, class_size));

    // Memcpy the entirety of the old class onto a new CPU heap pointer
    const struct MyriadObjectClass* class_cpy =
        (const struct MyriadObjectClass*) calloc(1, class_size);
    memcpy((void*)class_cpy, self, class_size);

    // Embedded object's class set to our GPU class; this ignores $clobber
    memcpy((void*)&class_cpy->_.m_class, &dev_class, sizeof(void*));

    CUDA_CHECK_RETURN(
        cudaMemcpy(
            (void*)dev_class,
            class_cpy,
            class_size,
            cudaMemcpyHostToDevice
            )
        );

    free((void*)class_cpy); // Can safely free since underclasses get nothing

    return (void*) dev_class;
    #else
    return NULL;
    #endif
        """

    @_myriadclass_method
    def cls_decudafy(self,
                     cuda_self: MyriadScalar("cuda_self", MVoid, ptr=True)):
        """
    fputs("De-CUDAfying a class is undefined behavior. Aborted. ", stderr);
    return;
        """

    @classmethod
    def gen_init_funs(cls):
        """ Generates the init* functions for modules as a big string """
        # Make temporary dictionary since we need to add an extra value
        tmp_dict = {
            "own_methods": getattr(cls, "own_methods"),
            "subclasses":  get_all_subclasses(cls)}
        template = MakoTemplate(INIT_OB_FUN_TEMPLATE, tmp_dict)
        LOG.debug("Rendering init functions for TODO")
        template.render()
        setattr(cls, "init_functions", template.buffer)

    @classmethod
    def _template_creator_helper(cls):
        """
        Initializes templates for the current class
        """
        # Create empty local namespace
        local_namespace = {}
        # Get values from class namespace
        own_methods = getattr(cls, "own_methods")
        cls_name = getattr(cls, "cls_name")
        obj_name = getattr(cls, "obj_name")
        obj_struct = getattr(cls, "obj_struct")
        # Initialize delegators/superdelegators in local namespace
        own_method_delgs = []
        for method in own_methods:
            own_method_delgs.append(
                (create_delegator(method, cls_name),
                 create_super_delegator(method, cls_name)))
        local_namespace["own_method_delgs"] = own_method_delgs
        # Fill local namespace with values we need for template rendering
        local_namespace["own_methods"] = getattr(cls, "own_methods")
        local_namespace["cls_name"] = getattr(cls, "cls_name")
        local_namespace["obj_name"] = getattr(cls, "obj_name")
        local_namespace["obj_struct"] = getattr(cls, "obj_struct")
        local_namespace["myriad_methods"] = getattr(cls, "myriad_methods")
        local_namespace["init_functions"] = getattr(cls, "init_functions")
        local_namespace["local_includes"] = getattr(cls, "local_includes")
        local_namespace["lib_includes"] = getattr(cls, "lib_includes")
        local_namespace["myriad_classes"] = MyriadMetaclass.myriad_classes
        local_namespace["our_subclasses"] = get_all_subclasses(cls)
        pprint(local_namespace["our_subclasses"])
        # Render main file templates
        setattr(cls, "cuh_file_template",
                MakoFileTemplate(
                    obj_name + ".cuh",
                    CUH_FILE_TEMPLATE,
                    local_namespace))
        LOG.debug("cuh_file_template done for %s", obj_name)
        setattr(cls, "cu_file_template",
                MakoFileTemplate(
                    obj_name + ".cu",
                    CU_FILE_TEMPLATE,
                    local_namespace))
        LOG.debug("cu_file_template done for %s", obj_name)
        # Initialize object struct conversion for CPython getter methods
        # Ignores superclass (_), class object, and array declarations
        # Places result in local namespace to avoid collisions/for efficiency
        pyc_scalars = {}
        for obj_var_name, obj_var_decl in obj_struct.members.items():
            if (not obj_var_name.startswith("_") and
                    not obj_var_name == "mclass" and
                    not isinstance(obj_var_decl.type, ArrayDecl)):
                pyc_scalars[obj_var_name] = c_decl_to_pybuildarg(obj_var_decl)
        local_namespace["pyc_scalar_types"] = pyc_scalars
        setattr(cls, "pyc_file_template",
                MakoFileTemplate(
                    "py" + obj_name.lower() + ".c",
                    PYC_COMP_FILE_TEMPLATE,
                    local_namespace))
        LOG.debug("pyc_file_template done for %s", obj_name)

    @classmethod
    def render_templates(cls, template_dir=None):
        """ Render internal templates to files"""
        # Get template rendering directory
        template_dir = template_dir if template_dir else os.getcwd()
        # Render templates for the superclass
        if cls is not MyriadObject:
            # Render init functions now that we have complete RTTI
            cls.gen_init_funs()
            getattr(cls.__bases__[0], "render_templates")()
        # Prepare templates for rendering by collecting subclass information
        cls._template_creator_helper()
        # Render templates for the current class
        LOG.debug("Rendering CUH File for %s", cls.__name__)
        getattr(cls, "cuh_file_template").render_to_file(overwrite=False)
        LOG.debug("Rendering CU file for %s", cls.__name__)
        getattr(cls, "cu_file_template").render_to_file(overwrite=False)
        # MyriadObject has its own special pyc/pyh files
        if cls is MyriadObject:
            LOG.debug("Rendering PYC File for MyriadObject")
            c_template = MakoFileTemplate("pymyriadobject.c",
                                          MYRIADOBJECT_PYC_FILE_TEMPLATE,
                                          cls.__dict__)
            c_template.render_to_file(overwrite=False)
            LOG.debug("Rendering PYH File for MyriadObject")
            h_template = MakoFileTemplate("pymyriadobject.h",
                                          MYRIADOBJECT_PYH_FILE_TEMPLATE,
                                          cls.__dict__)
            h_template.render_to_file(overwrite=False)
        else:
            LOG.debug("Rendering PYC File for %s", cls.__name__)
            getattr(cls, "pyc_file_template").render_to_file(overwrite=False)

    @classmethod
    def _fill_in_base_methods(cls,
                              child_namespace: OrderedDict,
                              myriad_methods: OrderedDict):
        """
        Fills in missing base methods (e.g. ctor/etc) in child's namespace.

        # TODO: Consider whether dtor/cudafy/etc. should be filled in
        """
        # Fill in ctor if it's missing
        if "ctor" not in myriad_methods:
            template = MakoTemplate(CTOR_TEMPLATE_TEMPLATE, child_namespace)
            LOG.debug("Rendering ctor template for %s",
                      child_namespace["obj_name"])
            template.render()
            myriad_methods["ctor"] = MyriadFunction.from_myriad_func(
                getattr(cls, "myriad_methods")["ctor"],
                fun_def=template.buffer)

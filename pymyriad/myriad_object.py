"""
Definition of the parent MyriadObject, from which all Myriad types inherit.
"""
import logging

from collections import OrderedDict
from pkg_resources import resource_string

from myriad_mako_wrapper import MakoTemplate, MakoFileTemplate

from myriad_types import MyriadScalar, MyriadFunction
from myriad_types import MVoid, MVarArgs, MInt

from myriad_metaclass import _myriadclass_method, _MyriadObjectBase
from myriad_metaclass import myriad_method_verbatim, MyriadMetaclass

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
    "templates/pymyriadobject.c.mako").decode("UTF-8")

MYRIADOBJECT_PYH_FILE_TEMPLATE = resource_string(
    __name__,
    "templates/pymyriadobject.h.mako").decode("UTF-8")

CTOR_TEMPLATE_TEMPLATE = resource_string(
    __name__,
    "templates/ctor_template.mako").decode("UTF-8")

CLS_CTOR_TEMPLATE = resource_string(
    __name__,
    "templates/class_ctor_template.mako").decode("UTF-8")

CLS_CUDAFY_TEMPLATE = resource_string(
    __name__,
    "templates/class_cudafy_template.mako").decode("UTF-8")


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
    def render_templates(cls):
        """ Render internal templates to files"""
        # Render templates for the superclass
        if cls is not MyriadObject:
            cls.__bases__[0].render_templates()
        # Render templates for the current class
        LOG.debug("Rendering H File for %s", cls.__name__)
        getattr(cls, "header_file_template").render_to_file(overwrite=False)
        LOG.debug("Rendering C File for %s", cls.__name__)
        getattr(cls, "c_file_template").render_to_file(overwrite=False)
        LOG.debug("Rendering CUH File for %s", cls.__name__)
        getattr(cls, "cuh_file_template").render_to_file(overwrite=False)
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
        if "cls_cudafy" not in myriad_methods:
            template = MakoTemplate(CLS_CUDAFY_TEMPLATE, child_namespace)
            LOG.debug("Rendering cls_cudafy template for %s",
                      child_namespace["obj_name"])
            template.render()
            myriad_methods["cls_cudafy"] = MyriadFunction.from_myriad_func(
                getattr(cls, "myriad_methods")["cls_cudafy"],
                fun_def=template.buffer)
        if "cls_ctor" not in myriad_methods:
            template = MakoTemplate(CLS_CTOR_TEMPLATE, child_namespace)
            LOG.debug("Rendering cls_ctor template for %s",
                      child_namespace["obj_name"])
            template.render()
            myriad_methods["cls_ctor"] = MyriadFunction.from_myriad_func(
                getattr(cls, "myriad_methods")["cls_ctor"],
                fun_def=template.buffer)

"""
Definition of the parent MyriadObject, from which all Myriad types inherit.
"""
import logging
import os
from collections import OrderedDict
from pkg_resources import resource_string
from pycparser.c_ast import ArrayDecl

from .myriad_mako_wrapper import MakoTemplate, MakoFileTemplate
from .myriad_types import MyriadScalar, MyriadFunction
from .myriad_types import MVoid, MVarArgs, MInt
from .myriad_types import c_decl_to_pybuildarg
from .myriad_metaclass import _MyriadObjectBase
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

CTOR_TEMPLATE = resource_string(
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
    def cudafy(self, cuda_self: MyriadScalar("cuda_self", MVoid, ptr=True)):
        """
    #ifdef CUDA
        // Memcpy entire struct to CUDA pointer
        CUDA_CHECK_CALL(
            cudaMemcpy(
                cuda_self,
                self,
                myriad_sizeof(self),
                cudaMemcpyHostToDevice));
    #else
        fputs("CUDAfication not supported when CUDA not enabled.\\n", stderr);
    #endif
        """

    @myriad_method_verbatim
    def decudafy(self, cuda_self: MyriadScalar("cuda_self", MVoid, ptr=True)):
        """
    #ifdef CUDA
        // Memcpy entire struct back from CUDA pointer
        CUDA_CHECK_CALL(
            cudaMemcpy(
                self,
                cuda_self,
                myriad_sizeof(self),
                cudaMemcpyDeviceToHost));
    #else
        fputs("CUDAfication not supported when CUDA not enabled.\\n", stderr);
    #endif
        """

    @classmethod
    def gen_init_funs(cls):
        """ Generates the init* functions for modules as a big string """
        # Make temporary dictionary since we need to add an extra value
        tmp_dict = {
            "own_methods": getattr(cls, "own_methods"),
            "our_subclasses":  get_all_subclasses(cls)}
        template = MakoTemplate(INIT_OB_FUN_TEMPLATE, tmp_dict)
        LOG.debug("Rendering init functions for TODO")
        template.render()
        setattr(cls, "init_functions", template.buffer)

    @classmethod
    def _template_creator_helper(cls, template_dir=None):
        """
        Initializes templates for the current class
        """
        # Set template directory
        template_dir = template_dir if template_dir else os.getcwd()
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
        if cls is not MyriadObject:
            local_namespace["super_obj_name"] = getattr(cls, "super_obj_name")
        else:
            local_namespace["super_obj_name"] = None
        # Render main file templates
        setattr(cls, "cuh_file_template",
                MakoFileTemplate(
                    os.path.join(template_dir, obj_name + ".cuh"),
                    CUH_FILE_TEMPLATE,
                    local_namespace))
        LOG.debug("cuh_file_template done for %s", obj_name)
        setattr(cls, "cu_file_template",
                MakoFileTemplate(
                    os.path.join(template_dir, obj_name + ".cu"),
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
                    os.path.join(template_dir, "py" + obj_name.lower() + ".c"),
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
            getattr(cls.__bases__[0], "render_templates")(template_dir)
        # Prepare templates for rendering by collecting subclass information
        cls._template_creator_helper(template_dir)
        # Render templates for the current class
        LOG.debug("Rendering CUH File for %s", cls.__name__)
        getattr(cls, "cuh_file_template").render_to_file(overwrite=False)
        LOG.debug("Rendering CU file for %s", cls.__name__)
        getattr(cls, "cu_file_template").render_to_file(overwrite=False)
        # MyriadObject has its own special pyc/pyh files
        if cls is MyriadObject:
            LOG.debug("Rendering PYC File for MyriadObject")
            c_template = MakoFileTemplate(
                os.path.join(template_dir, "pymyriadobject.c"),
                MYRIADOBJECT_PYC_FILE_TEMPLATE,
                cls.__dict__)
            c_template.render_to_file(overwrite=False)
            LOG.debug("Rendering PYH File for MyriadObject")
            h_template = MakoFileTemplate(
                os.path.join(template_dir, "pymyriadobject.h"),
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
            template = MakoTemplate(CTOR_TEMPLATE, child_namespace)
            LOG.debug("Rendering ctor template for %s",
                      child_namespace["obj_name"])
            template.render()
            myriad_methods["ctor"] = MyriadFunction.from_myriad_func(
                getattr(cls, "myriad_methods")["ctor"],
                fun_def=template.buffer)

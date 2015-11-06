"""
.. module:: myriad_metaclass
    :platform: Linux
    :synposis: Provides metaclass for automatic Myriad integration

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>

"""
import inspect
import logging

from collections import OrderedDict
from copy import copy
from functools import wraps

from pkg_resources import resource_string

from myriad_mako_wrapper import MakoTemplate, MakoFileTemplate

from myriad_utils import OrderedSet

from myriad_types import MyriadScalar, MyriadFunction, MyriadStructType
from myriad_types import _MyriadBase, MyriadCType, MyriadTimeseriesVector
from myriad_types import MDouble, MVoid

from ast_function_assembler import pyfun_to_cfun

#######
# Log #
#######

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

#############
# Constants #
#############

# Default include headers for all C files
DEFAULT_LIB_INCLUDES = {"stdlib.h",
                        "stdio.h",
                        "assert.h",
                        "string.h",
                        "stddef.h",
                        "stdarg.h",
                        "stdint.h"}

# Default include headers for CUDA files
DEFAULT_CUDA_INCLUDES = {"cuda_runtime.h", "cuda_runtime_api.h"}


#############
# Templates #
#############

DELG_TEMPLATE = resource_string(
    __name__,
    "templates/delegator_func.mako").decode("UTF-8")

SUPER_DELG_TEMPLATE = resource_string(
    __name__,
    "templates/super_delegator_func.mako").decode("UTF-8")

CLS_CTOR_TEMPLATE = resource_string(
    __name__,
    "templates/class_ctor_template.mako").decode("UTF-8")

CLS_CUDAFY_TEMPLATE = resource_string(
    __name__,
    "templates/class_cudafy_template.mako").decode("UTF-8")

HEADER_FILE_TEMPLATE = resource_string(
    __name__,
    "templates/header_file.mako").decode("UTF-8")

CUH_FILE_TEMPLATE = resource_string(
    __name__,
    "templates/cuda_header_file.mako").decode("UTF-8")

C_FILE_TEMPLATE = resource_string(
    __name__,
    "templates/c_file.mako").decode("UTF-8")

PYC_COMP_FILE_TEMPLATE = resource_string(
    __name__,
    "templates/pyc_file.mako").decode("UTF-8")

######################
# Delegator Creation #
######################


def create_delegator(instance_fxn: MyriadFunction,
                     classname: str) -> MyriadFunction:
    """
    Creates a delegator function based on a function definition.

    :param MyriadFunction instance_fxn: Instance function to be wrapped
    :param str classname: Name of the class delegator
    :return: New MyriadFunction representing the delegator around instance_fxn
    :rtype: MyriadFunction
    """
    # Create copy with modified identifier
    ist_cpy = MyriadFunction.from_myriad_func(
        instance_fxn,
        ident=instance_fxn.ident.partition(classname + "_")[-1])
    # Generate template and render into copy's definition
    template_vars = {"delegator": ist_cpy, "classname": classname}
    template = MakoTemplate(DELG_TEMPLATE, template_vars)
    LOG.debug("Rendering create_delegator template for %s", classname)
    template.render()
    ist_cpy.fun_def = template.buffer
    # Return created copy
    return ist_cpy


def create_super_delegator(delg_fxn: MyriadFunction,
                           classname: str) -> MyriadFunction:
    """
    Create super delegator function.

    :param MyriadFunction delg_fxn: Delegator to create super_* wrapper for
    :param str classname: Name of the base class for this super delegator

    :return: Super delegator method as a MyriadFunction
    :rtype: MyriadFunction
    """
    # Create copy of delegator function with modified parameters
    super_args = copy(delg_fxn.args_list)
    super_class_arg = MyriadScalar("_class", MVoid, True, ["const"])
    tmp_arg_indx = len(super_args) + 1
    super_args[tmp_arg_indx] = super_class_arg
    super_args.move_to_end(tmp_arg_indx, last=False)
    s_delg_f = MyriadFunction.from_myriad_func(delg_fxn,
                                               "super_" + delg_fxn.ident,
                                               super_args)

    # Generate template and render
    template_vars = {"delegator": delg_fxn,
                     "super_delegator": s_delg_f,
                     "classname": classname}
    template = MakoTemplate(SUPER_DELG_TEMPLATE, template_vars)
    LOG.debug("Rendering create_super_delegator template for %s", classname)
    template.render()

    # Add rendered definition to function
    s_delg_f.fun_def = template.buffer
    return s_delg_f


def gen_instance_method_from_str(delegator, m_name: str,
                                 method_body: str) -> MyriadFunction:
    """
    Automatically generate a MyriadFunction wrapper for a method body.

    :param str m_name: Name to prepend to the instance method identifier
    :param str method_body: String template to use as the method body

    :return: Instance method as a MyriadFunction
    :rtype: MyriadFunction
    """
    return MyriadFunction(m_name + '_' + delegator.ident,
                          args_list=delegator.args_list,
                          ret_var=delegator.ret_var,
                          storage=['static'],
                          fun_def=method_body)

#####################
# Method Decorators #
#####################


def myriad_method(method):
    """
    Tags a method in a class to be a myriad method (i.e. converted to a C func)
    NOTE: This MUST be the first decorator applied to the function! E.g.:

        @another_decorator
        @yet_another_decorator
        @myriad_method
        def my_fn(stuff):
            pass

    This is because decorators replace the wrapped function's signature.
    """
    @wraps(method)
    def inner(*args, **kwargs):
        """ Dummy inner function to prevent direct method calls """
        raise Exception("Cannot directly call a myriad method")
    LOG.debug("myriad_method annotation wrapping %s", method.__name__)
    setattr(inner, "is_myriad_method", True)
    setattr(inner, "original_fun", method)
    return inner


def myriad_method_verbatim(method):
    """
    Tags a method in a class to be a myriad method (i.e. converted to a C func)
    but takes the docstring as verbatim C code.

    NOTE: This MUST be the first decorator applied to the function! E.g.:

        @another_decorator
        @yet_another_decorator
        @myriad_method_verbatim
        def my_fn(stuff):
            pass

    This is because decorators replace the wrapped function's signature.
    """
    @wraps(method)
    def inner(*args, **kwargs):
        """ Dummy inner function to prevent direct method calls """
        raise Exception("Cannot directly call a myriad method")
    setattr(inner, "is_myriad_method_verbatim", True)
    setattr(inner, "is_myriad_method", True)
    setattr(inner, "original_fun", method)
    return inner

#####################
# MetaClass Wrapper #
#####################


class _MyriadObjectBase(object):
    """ Dummy placeholder class used for type checking """
    pass


def _method_organizer_helper(supercls: _MyriadObjectBase,
                             myriad_methods: OrderedDict,
                             myriad_cls_vars: OrderedDict,
                             verbatim_methods: OrderedSet=None) -> OrderedSet:
    """
    Organizes Myriad Methods, including inheritance and verbatim methods.

    Verbatim methods are converted differently than pythonic methods; their
    docstring is embedded 'verbatim' into the template instead of going through
    the AST conversion.

    Returns an OrderedSet of methods not defined in the superclass
    # TODO: Make sure own_methods aren't also defined in super-superclass/etc
    """
    # Convert methods; remember, items() returns a read-only view
    for m_ident, method in myriad_methods.items():
        if verbatim_methods is not None and m_ident in verbatim_methods:
            if method.__doc__ is None or method.__doc__ == "":
                raise Exception("Verbatim method cannot have empty docstring")
            myriad_methods[m_ident] = method.__doc__
        else:
            myriad_methods[m_ident] = pyfun_to_cfun(method)

    # Inherit parent Myriad Methods
    for super_ident, super_method in supercls.myriad_methods.items():
        # ... if we haven't provided our own (overwriting)
        if super_ident not in myriad_methods:
            myriad_methods[super_ident] = super_method

    # Get a set difference between super/own methods for class struct
    super_methods_ident_set = OrderedSet(
        [(k, v) for k, v in supercls.myriad_methods.items()])
    LOG.debug("_method_organizer_helper super methods found: %r",
              super_methods_ident_set)
    all_methods_ident_set = OrderedSet(
        [(k, v) for k, v in myriad_methods.items()])
    own_methods = all_methods_ident_set - super_methods_ident_set
    LOG.debug("_method_organizer_helper own methods identified: %r",
              own_methods)

    # Struct definition representing class methods
    for _, method in own_methods:
        new_ident = "my_" + method.fun_typedef.name
        m_scal = MyriadScalar(new_ident, method.base_type)
        myriad_cls_vars[new_ident] = m_scal
    LOG.debug("_method_organizer_helper class variables selected: %r",
              myriad_cls_vars)

    return own_methods


def _template_creator_helper(namespace: OrderedDict) -> OrderedDict:
    """
    Creates templates using namespace, and returns the updated namespace.
    """
    namespace["c_file_template"] = MakoFileTemplate(
        namespace["obj_name"] + ".c",
        C_FILE_TEMPLATE,
        namespace)
    namespace["header_file_template"] = MakoFileTemplate(
        namespace["obj_name"] + ".h",
        HEADER_FILE_TEMPLATE,
        namespace)
    namespace["cuh_file_template"] = MakoFileTemplate(
        namespace["obj_name"] + ".cuh",
        CUH_FILE_TEMPLATE,
        namespace)
    # TODO: CU file template
    namespace["pyc_file_template"] = MakoFileTemplate(
        "py_" + namespace["obj_name"] + ".c",
        PYC_COMP_FILE_TEMPLATE,
        namespace)
    return namespace


def _generate_includes_helper(superclass, features: set=None) -> (set, set):
    """ Generates local and lib includes based on superclass and features """
    local_includes = [superclass.__name__]
    lib_includes = copy(DEFAULT_LIB_INCLUDES)
    # TODO: Add CUDA includes on-demand
    return (local_includes, lib_includes)


def _parse_namespace(namespace: dict,
                     name: str,
                     myriad_methods: OrderedDict,
                     myriad_obj_vars: OrderedDict,
                     verbatim_methods: OrderedSet):
    """
    Parses the given namespace, updates the last three input arguments to have:
        1) OrderedDict of myriad_methods
        2) OrderedDict of myriad_obj_vars
        3) OrderedSet of verbatim methods
    """
    # Extracts variables and myriad methods from class definition
    for k, val in namespace.items():
        # if val is ...
        # ... a registered myriad method
        if hasattr(val, "is_myriad_method"):
            LOG.debug("%s is a myriad method in %s", k, name)
            myriad_methods[k] = val.original_fun
            # Verbatim methods are tracked in a set
            if hasattr(val, "is_myriad_method_verbatim"):
                LOG.debug("%s is a verbatim myriad method in %s", k, name)
                verbatim_methods.add(val)
        # ... some generic non-Myriad function or method
        elif inspect.isfunction(val) or inspect.ismethod(val):
            LOG.debug("%s is a function or method, ignoring for %s", k, name)
        # ... some generic instance of a _MyriadBase type
        elif issubclass(val.__class__, _MyriadBase):
            LOG.debug("%s is a Myriad-type non-function attribute", k)
            myriad_obj_vars[k] = val
            LOG.debug("%s was added as an object variable to %s", k, name)
        # ... a type statement of base type MyriadCType (e.g. MDouble)
        elif issubclass(val.__class__, MyriadCType):
            myriad_obj_vars[k] = MyriadScalar(k, val)
            LOG.debug("%s has decl %s", k, myriad_obj_vars[k].stringify_decl())
            LOG.debug("%s was added as an object variable to %s", k, name)
        # ... a timeseries variable
        elif val is MyriadTimeseriesVector:
            # TODO: Enable different precisions for MyriadTimeseries
            myriad_obj_vars[k] = MyriadScalar(k, MDouble, arr_id="SIMUL_LEN")
        # ... a python meta value (e.g.  __module__) we shouldn't mess with
        elif k.startswith("__"):
            LOG.debug("Built-in method %r ignored for %s", k, name)
        # TODO: Figure out other valid values for namespace variables
        else:
            LOG.info("Unsupported var type for %r, ignoring in %s", k, name)
    LOG.debug("myriad_obj_vars for %s: %s ", name, myriad_obj_vars)


def _init_module_vars(obj_name: str,
                      cls_name: str,
                      is_myriad_obj: bool) -> OrderedDict:
    """ Special method for initializing MyriadObject objects"""
    # TODO: Make this templatable (for myriad_* methods)
    module_vars = OrderedDict()
    if is_myriad_obj:
        module_vars['object'] = """
static struct MyriadClass object[] =
{
    {
        { object + 1 },
        object,
        NULL,
        sizeof(struct MyriadObject),
        MyriadObject_myriad_ctor,
        MyriadObject_myriad_dtor,
        MyriadObject_myriad_cudafy,
        MyriadObject_myriad_decudafy,
    },
    {
        { object + 1 },
        object,
        NULL,
        sizeof(struct MyriadClass),
        MyriadClass_myriad_ctor,
        MyriadClass_myriad_dtor,
        MyriadClass_myriad_cudafy,
        MyriadClass_myriad_decudafy,
    }
};
        """
    module_vars[obj_name] =\
        MyriadScalar(
            obj_name,
            MVoid,
            True,
            quals=["const"],
            init="object" if is_myriad_obj else None)
    module_vars[cls_name] =\
        MyriadScalar(
            cls_name,
            MVoid,
            True,
            quals=["const"],
            init="object + 1" if is_myriad_obj else None)
    return module_vars


class MyriadMetaclass(type):
    """
    TODO: Documentation for MyriadMetaclass
    """

    @classmethod
    def __prepare__(mcs, name, bases):
        """
        Force the class to use an OrderedDict as its __dict__, for purposes of
        enforcing strict ordering of keys (necessary for making structs).
        """
        return OrderedDict()

    @staticmethod
    def myriad_init(self, **kwargs):
        # TODO: Check if all kwargs (including parents) are set
        for argname, argval in kwargs.items():
            self.__setattr__(argname, argval)

    @staticmethod
    def myriad_set_attr(self, **kwargs):
        """
        Prevent users from accessing objects except through py_x interfaces
        """
        raise NotImplementedError("Cannot set object attributes (yet).")

    def __new__(mcs, name, bases, namespace, **kwds):
        if len(bases) > 1:
            raise NotImplementedError("Multiple inheritance is not supported.")

        # Check if the class inherits from MyriadObject
        if not issubclass(bases[0], _MyriadObjectBase):
            raise TypeError("Myriad modules must inherit from MyriadObject")
        supercls = bases[0]  # Alias for base class

        # Setup object/class variables, methods, and verbatim methods
        myriad_cls_vars = OrderedDict()
        myriad_obj_vars = OrderedDict()
        myriad_methods = OrderedDict()
        verbatim_methods = set()

        # Setup object with implicit superclass to start of struct definition
        if supercls is not _MyriadObjectBase:
            myriad_obj_vars["_"] = supercls.obj_struct("_", quals=["const"])
            myriad_cls_vars["_"] = supercls.cls_struct("_", quals=["const"])

        # Parse namespace into appropriate variables
        _parse_namespace(namespace,
                         name,
                         myriad_methods,
                         myriad_obj_vars,
                         verbatim_methods)

        # Object Name and Class Name are automatically derived from name
        namespace["obj_name"] = name
        namespace["cls_name"] = name + "Class"

        # Struct definition representing object state
        namespace["obj_struct"] = MyriadStructType(namespace["obj_name"],
                                                   myriad_obj_vars)

        # Organize myriad methods and class struct members
        if supercls is not _MyriadObjectBase:
            namespace["own_methods"] = _method_organizer_helper(
                supercls,
                myriad_methods,
                myriad_cls_vars,
                verbatim_methods)
            namespace["local_includes"], namespace["lib_includes"] = \
                _generate_includes_helper(supercls)

        # Create myriad class struct
        namespace["cls_struct"] = MyriadStructType(namespace["cls_name"],
                                                   myriad_cls_vars)

        # Add other objects to namespace
        namespace["myriad_methods"] = myriad_methods
        namespace["myriad_obj_vars"] = myriad_obj_vars
        namespace["myriad_cls_vars"] = myriad_cls_vars

        # Initialize module variables
        namespace["myriad_module_vars"] =\
            _init_module_vars(
                namespace["obj_name"],
                namespace["cls_name"],
                supercls is _MyriadObjectBase)

        # TODO: Initialize module functions

        # Write templates now that we have full information
        LOG.debug("Creating templates for class %s", name)
        namespace = _template_creator_helper(namespace)

        # Finally, delete function from namespace
        for method_id in myriad_methods.keys():
            del namespace[method_id]

        # Generate internal module representation
        namespace["__init__"] = MyriadMetaclass.myriad_init
        namespace["__setattr__"] = MyriadMetaclass.myriad_set_attr
        return type.__new__(mcs, name, (supercls,), dict(namespace))


# TODO: MyriadObject definition
class MyriadObject(_MyriadObjectBase, metaclass=MyriadMetaclass):
    """ Base class that every myriad object inherits from """

    @classmethod
    def render_templates(cls):
        """ Render internal templates to files"""
        LOG.debug("Rendering H File for %s", cls.__name__)
        getattr(cls, "header_file_template").render_to_file()
        LOG.debug("Rendering C File for %s", cls.__name__)
        getattr(cls, "c_file_template").render_to_file()
        LOG.debug("Rendering CUH File for %s", cls.__name__)
        getattr(cls, "cuh_file_template").render_to_file()
        LOG.debug("Rendering PYC File for %s", cls.__name__)
        getattr(cls, "pyc_file_template").render_to_file()

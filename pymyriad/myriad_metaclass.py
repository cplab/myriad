"""
.. module:: myriad_metaclass
    :platform: Linux
    :synposis: Provides metaclass for automatic Myriad integration

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>


"""
import inspect

from collections import OrderedDict
from copy import copy
from functools import wraps
from warnings import warn

from myriad_mako_wrapper import MakoTemplate

from myriad_utils import OrderedSet

from myriad_types import MyriadScalar, MyriadFunction, MyriadStructType
from myriad_types import MVoid, _MyriadBase, MyriadCType

from ast_function_assembler import pyfun_to_cfun

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

DELG_TEMPLATE = open("templates/delegator_func.mako", 'r').read()

SUPER_DELG_TEMPLATE = open("templates/super_delegator_func.mako", 'r').read()

CLS_CTOR_TEMPLATE = open("templates/class_ctor_template.mako", 'r').read()

CLS_CUDAFY_TEMPLATE = open("templates/class_cudafy_template.mako", 'r').read()


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
        """Dummy inner function to prevent direct method calls"""
        raise Exception("Cannot directly call a myriad method")
    inner.__dict__["is_myriad_method"] = True
    inner.__dict__["original_fun"] = method
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
        """Dummy inner function to prevent direct method calls"""
        raise Exception("Cannot directly call a myriad method")
    inner.__dict__["is_myriad_method_verbatim"] = True
    inner.__dict__["is_myriad_method"] = True
    inner.__dict__["original_fun"] = method
    return inner

#####################
# MetaClass Wrapper #
#####################


# Dummy class used for type checking
class _MyriadObjectBase(object):
    pass


def _method_organizer_helper(
        myriad_methods: OrderedDict,
        supercls: _MyriadObjectBase,
        myriad_cls_vars: OrderedDict,
        verbatim_methods: set=None) -> (OrderedDict, OrderedDict):
    """
    # TODO: Better documentation of method_organizer_helper
    Organizes Myriad Methods, including inheritance.

    Verbatim methods are converted differently than pythonic methods.

    Returns a tuple of the class struct variables, and the myriad methods.
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
    all_methods_ident_set = OrderedSet(
        [(k, v) for k, v in myriad_methods.items()])
    own_methods = all_methods_ident_set - super_methods_ident_set

    # Struct definition representing class methods
    for _, method in own_methods:
        new_ident = "my_" + method.fun_typedef.name
        m_scal = MyriadScalar(new_ident, method.base_type)
        myriad_cls_vars[new_ident] = m_scal

    return (myriad_cls_vars, myriad_methods)


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

        # Setup methods and variables as ordered dictionaries
        myriad_methods = OrderedDict()
        myriad_obj_vars = OrderedDict()
        myriad_cls_vars = OrderedDict()
        verbatim_methods = set()

        # Setup object with implicit superclass to start of struct definition
        if supercls is not _MyriadObjectBase:
            myriad_obj_vars["_"] = supercls.obj_struct("_", quals=["const"])
            myriad_cls_vars["_"] = supercls.cls_struct("_", quals=["const"])

        # Extracts variables and myriad methods from class definition
        for k, val in namespace.items():
            # if val is ...
            # ... a registered myriad method
            if hasattr(val, "is_myriad_method"):
                # print(k + " is a myriad method")
                myriad_methods[k] = val.original_fun
                # Verbatim methods are tracked in a set
                if hasattr(val, "is_myriad_method_verbatim"):
                    verbatim_methods.add(k)
            # ... some generic non-Myriad function or method
            elif inspect.isfunction(val) or inspect.ismethod(val):
                # print(k + " is a function or method, ignoring...")
                pass
            # ... some generic instance of a _MyriadBase type
            elif issubclass(val.__class__, _MyriadBase):
                myriad_obj_vars[k] = val
                # print(k + " is a Myriad-type non-function attribute")
            # ... a type statement of base type MyriadCType (e.g. MDouble)
            elif issubclass(val.__class__, MyriadCType):
                # TODO: Better type detection here for corner cases (e.g. ptr)
                myriad_obj_vars[k] = MyriadScalar(k, val)
                # print(k + " has decl " + myriad_obj_vars[k].stringify_decl())
            # ... a python meta value (e.g.  __module__) we shouldn't mess with
            elif k.startswith("__"):
                pass
            # TODO: Figure out other valid values for namespace variables
            else:
                warn("Unsupported variable type for " + k)

        # Object Name and Class Name are automatically derived from name
        namespace["obj_name"] = name
        namespace["cls_name"] = name + "Class"

        # Struct definition representing object state
        namespace["obj_struct"] = MyriadStructType(namespace["obj_name"],
                                                   myriad_obj_vars)

        # Organize myriad methods and class struct members
        if supercls is not _MyriadObjectBase:
            myriad_cls_vars, myriad_methods = _method_organizer_helper(
                myriad_methods,
                supercls,
                myriad_cls_vars)

        # Create myriad class struct
        namespace["cls_struct"] = MyriadStructType(namespace["cls_name"],
                                                   myriad_cls_vars)

        # Add other objects to namespace
        namespace["myriad_methods"] = myriad_methods
        namespace["myriad_obj_vars"] = myriad_obj_vars
        namespace["myriad_cls_vars"] = myriad_cls_vars

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
    pass

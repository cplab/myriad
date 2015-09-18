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

from myriad_types import MyriadScalar, MyriadFunction
from myriad_types import MVoid, _MyriadBase, MyriadCType

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

####################
# Method Decorator #
####################


def myriad_method(method):
    """
    NOTE: This MUST be the first decorator applied to the function! E.g.:

        @another_decorator
        @yet_another_decorator
        @print_class_name
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

#####################
# MetaClass Wrapper #
#####################


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

        # TODO: Check if the class inherits from MyriadObject
        # if not issubclass(bases[0], MyriadObject):
        #     raise TypeError("Myriad modules must inherit from MyriadObject")

        myriad_methods = OrderedDict()
        myriad_vars = OrderedDict()

        # Extracts variables and myriad methods from class definition
        for k, val in namespace.items():
            # if val is ...
            # ... a registered myriad method
            if hasattr(val, "is_myriad_method"):
                print(k + " is a myriad method")
                myriad_methods[k] = val.original_fun
            # ... some generic non-Myriad function or method
            elif inspect.isfunction(val) or inspect.ismethod(val):
                print(k + " is a function or method")
            # ... some generic instance of a _MyriadBase type
            elif issubclass(val.__class__, _MyriadBase):
                myriad_vars[k] = val
                print(k + " is a Myriad-type non-function attribute")
            # ... a type statement of base type MyriadCType (e.g. MDouble)
            elif issubclass(val.__class__, MyriadCType):
                # TODO: Better type detection here for corner cases (e.g. ptr)
                myriad_vars[k] = MyriadScalar(k, val)
                print(k + " has decl: " + myriad_vars[k].stringify_decl())
            # TODO: Figure out other valid values for namespace variables
            else:
                warn("Unsupported variable type for " + k)

        # Finally, delete function from namespace
        for method_id in myriad_methods.keys():
            del namespace[method_id]

        # Generate internal module representation
        namespace["__init__"] = MyriadMetaclass.myriad_init
        namespace["__setattr__"] = MyriadMetaclass.myriad_set_attr
        result = type.__new__(mcs, name, (object,), dict(namespace))
        return result

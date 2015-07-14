"""
TODO: Docstring
"""

from collections import OrderedDict
from functools import wraps
import inspect
from warnings import warn

import myriad_types
from myriad_module import MyriadModule
from MyriadObject import MyriadObject


class MyriadMeta(type):
    """
    Metaclass for intercepting MyriadObject child declarations and doing
    necessary 'behind the scenes' work.
    """

    @classmethod
    def __prepare__(mcs, name, bases):
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

    def __new__(metacls, name, bases, namespace, **kwds):
        if len(bases) > 1:
            raise NotImplementedError("Multiple inheritance is not supported.")

        # Check if the class inherits from MyriadObject
        if not issubclass(bases[0], MyriadObject):
            raise TypeError("Myriad modules must inherit from MyriadObject")

        myriad_methods = OrderedDict()
        myriad_vars = OrderedDict()

        # Extracts variables and myriad methods from class definition
        for k, v in namespace.items():
            # if v is ...
            # ... a registered myriad method
            if hasattr(v, "is_myriad_method"):
                print(k + " is a myriad method")
                myriad_methods[k] = v.original_fun
            # ... some generic non-Myriad function or method
            elif inspect.isfunction(v) or inspect.ismethod(v):
                print(k + " is a function or method")
            # ... some generic instance of a _MyriadBase type
            elif issubclass(v.__class__, myriad_types._MyriadBase):
                myriad_vars[k] = v
                print(k + " is a Myriad-type non-function attribute")
            # ... a type statement of base type MyriadCType (e.g. MDouble)
            elif issubclass(v.__class__, myriad_types.MyriadCType):
                # TODO: Better type detection here for corner cases (e.g. ptr)
                myriad_vars[k] = myriad_types.MyriadScalar(k, v)
                print(k + " has decl: " + myriad_vars[k].stringify_decl())
            # TODO: Figure out other valid values for namespace variables
            else:
                warn("Unsupported variable type for " + k)

        # Finally, delete function from namespace
        for method_id in myriad_methods.keys():
            del namespace[method_id]

        # Generate internal module representation
        namespace["_my_module"] = MyriadModule(bases[0]._my_module,
                                               name,
                                               obj_vars=myriad_vars,
                                               methods=myriad_methods)
        namespace["__init__"] = MyriadMeta.myriad_init
        namespace["__setattr__"] = MyriadMeta.myriad_set_attr
        result = type.__new__(metacls, name, (object,), dict(namespace))
        # result.__init__ = MyriadMeta.my_initx
        return result


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
        raise Exception("Cannot directly call a myriad method")
    inner.__dict__["is_myriad_method"] = True
    inner.__dict__["original_fun"] = method
    return inner


class Soma(MyriadObject, metaclass=MyriadMeta):
    """TODO"""
    capacitance = myriad_types.MDouble
    vm = myriad_types.MyriadScalar("vm", myriad_types.MDouble, ptr=True)

    @myriad_method
    def zardoz(self,
               a: myriad_types.MInt,
               b: myriad_types.MInt) -> myriad_types.MInt:
        c = 0
        if a > b:
            a = b
            c = a + b
        return c

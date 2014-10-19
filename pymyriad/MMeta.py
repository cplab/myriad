"""
TODO: Docstring
"""

from collections import OrderedDict
import inspect

import myriad_types


class MyriadMeta(type):

    @classmethod
    def __prepare__(mcl, name, bases):
        return OrderedDict()

    def __new__(metacls, name, bases, namespace, **kwds):
        result = type.__new__(metacls, name, bases, dict(namespace))
        for k,v in namespace.items():
            print(k)
            if inspect.ismethod(v):
                print("True")
            else:
                print("False")
        # result.__init__ = None
        return result

def myriad_method(cuda=False):
    def wrap(f):
        def wrapped_f(*args):
            print("lol no function for you")
            print(inspect.signature(f))
            print(inspect.ismethod(f))
        return wrapped_f
    return wrap

class Soma(object, metaclass=MyriadMeta):

    vm = myriad_types.MyriadScalar("vm", myriad_types.MDouble, ptr=True)
    capacitance = myriad_types.MyriadScalar("capacitance", myriad_types.MDouble)

    @myriad_method(cuda=False)
    def my_fun(self,
               a: myriad_types.MyriadScalar("a", myriad_types.MInt),
               b: myriad_types.MyriadScalar("b", myriad_types.MInt)):
        if a > b:
            a = b
            c = a + b
        return c

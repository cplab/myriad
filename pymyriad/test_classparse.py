"""
"""
from ast_function_assembler import pyfun_to_cfun
from myriad_types import MInt
from inspect import getsource
import re


class TestClass(object):

    x = 12

    def __init__(self):
        pass

    def other_method(self, a):
        return self.x + a

    def test_method(self, a: MInt, b: MInt) -> MInt:
        self.x = self.other_method(a)
        return self.x


def dummy_fun(fun):
    fun_body = getsource(fun)
    if re.compile(r".+\.").match(fun.__qualname__) is not None:
        repl = "((struct " + fun.__qualname__.split('.')[0] + "*) self)->"
        fun_body = fun_body.replace("self.", repl)
    return fun_body

if __name__ == "__main__":
    m_fun = pyfun_to_cfun(TestClass.test_method, indent_lvl=2)
    print(m_fun.stringify_decl())
    print(m_fun.stringify_def())

"""
Tester module for parsing whole-classes
"""
import unittest
import logging

from myriad_testing import MyriadTestCase, set_external_loggers

from context import myriad
from myriad import ast_function_assembler
from myriad import myriad_types


#: Log for purposes of logging MyriadTestCase output
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class TestClass(object):
    """ Dummy test class """
    x = 12

    def __init__(self):
        pass

    def other_method(self, a):
        return self.x + a

    def test_method(self, a: myriad_types.MInt,
                    b: myriad_types.MInt) -> myriad_types.MInt:
        self.x = self.other_method(a)
        return self.x


@set_external_loggers("TestClassParsing", LOG)
class TestClassParsing(MyriadTestCase):
    """ Tests class parsing """

    def test_parse_class_method_decl(self):
        """ Testing if parsing class method declarations works """
        m_fun = ast_function_assembler.pyfun_to_cfun(TestClass.test_method)
        expected_decl = "int64_t test_method(void *self, int64_t a, int64_t b)"
        self.assertTrimStrEquals(m_fun.stringify_decl(), expected_decl)

    def test_parse_class_method_def(self):
        """ Testing if parsing class method definition works """
        m_fun = ast_function_assembler.pyfun_to_cfun(TestClass.test_method)
        expected_def = """
        ((struct TestClass*) self)->x = other_method(self, a);;
        return ((struct TestClass*) self)->x;
        """
        self.assertTrimStrEquals(m_fun.stringify_def(), expected_def)


if __name__ == "__main__":
    unittest.main()

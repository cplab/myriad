"""
Tester module for AST Function Assembler
Author: Pedro Rittner
"""

import unittest
import logging
from myriad_testing import MyriadTestCase, set_external_loggers

from context import myriad
from myriad import ast_function_assembler as ast_func
from myriad import myriad_types

#: Log for purposes of logging MyriadTestCase output
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


@set_external_loggers("ASTFunctionTester", LOG)
class ASTFunctionTester(MyriadTestCase):
    """
    Tests if AST function assembling works correctly
    """

    def test_simple_func_decl(self):
        """ Testing if parsing a simple function's declaration works """
        def test_fun(a: myriad_types.MInt,
                     b: myriad_types.MInt) -> myriad_types.MInt:
            return 0
        mfun = ast_func.pyfun_to_cfun(test_fun)
        self.assertIsNotNone(mfun)
        expected_decl = "int64_t test_fun(int64_t a, int64_t b)"
        self.assertTrimStrEquals(mfun.stringify_decl(), expected_decl)

    def test_simple_func_def(self):
        """ Tests if parsing a simple function's definition works """
        def test_fun(a: myriad_types.MInt,
                     b: myriad_types.MInt) -> myriad_types.MInt:
            x = 0
            while x < 3:
                if a == b:
                    x = x + 1
            return x
        mfun = ast_func.pyfun_to_cfun(test_fun)
        expected_def = """
        int_fast32_t x;
        x = 0;
        while (x < 3)
        {
            if (a == b)
            {
                x = x + 1;
            }
        }
        return x;
        """
        self.assertTrimStrEquals(mfun.stringify_def(), expected_def)

if __name__ == '__main__':
    unittest.main()

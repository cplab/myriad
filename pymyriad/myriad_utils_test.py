"""
Test cases for myriad_utils.

TODO: Write my tests for myriad_utils
"""

__author__ = ["Pedro Rittner"]

import unittest
import inspect

import myriad_utils


class TestTypeEnforcer(unittest.TestCase):
    """
    Test cases for TypeEnforcer.
    """

    def test_type_enforcer(self):
        """ Tests TypeEnforcer functionality """

        class _Foo(object, metaclass=myriad_utils.TypeEnforcer):
            def __init__(self, myint: int=0):
                self.myint = myint

        self.assertRaises(TypeError, _Foo, 5.)


class TestRemoveHeaderParens(unittest.TestCase):
    """
    Test cases for the function remove_header_parens
    """

    def test_remove_header_parens_1l(self):
        """ Tests removing a single-line header from a function. """
        def inner_fun(a_var: int) -> str:
            pass
        sourcelines = inspect.getsourcelines(inner_fun)[0]
        remains = myriad_utils.remove_header_parens(sourcelines)
        self.assertEqual(len(remains), 1)
        self.assertEqual(remains[0], '            pass\n')

    def test_remove_header_parens_2l(self):
        """ Tests removing a two-line header from a function."""
        def inner_fun(a_var: int,
                      b_var: int) -> str:
            pass
        sourcelines = inspect.getsourcelines(inner_fun)[0]
        remains = myriad_utils.remove_header_parens(sourcelines)
        self.assertEqual(len(remains), 1)
        self.assertEqual(remains[0], '            pass\n')


if __name__ == "__main__":
    unittest.main()

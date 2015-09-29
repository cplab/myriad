"""
Test cases for myriad_utils.

TODO: Write my tests for myriad_utils
"""

__author__ = ["Pedro Rittner"]

import unittest
import inspect

from myriad_utils import TypeEnforcer, OrderedSet, remove_header_parens


class TestTypeEnforcer(unittest.TestCase):
    """ Test cases for TypeEnforcer """

    def test_type_enforcer(self):
        """ Testing TypeEnforcer functionality """
        class _Foo(object, metaclass=TypeEnforcer):
            def __init__(self, myint: int=0):
                self.myint = myint

        self.assertRaises(TypeError, _Foo, 5.)


class TestRemoveHeaderParens(unittest.TestCase):
    """ Test cases for the function remove_header_parens """

    def test_remove_header_parens_1l(self):
        """ Testing removing a single-line header from a function """
        def inner_fun(a_var: int) -> str:
            pass
        sourcelines = inspect.getsourcelines(inner_fun)[0]
        remains = remove_header_parens(sourcelines)
        self.assertEqual(len(remains), 1)
        self.assertEqual(remains[0], '            pass\n')

    def test_remove_header_parens_2l(self):
        """ Testing removing a two-line header from a function """
        def inner_fun(a_var: int,
                      b_var: int) -> str:
            pass
        sourcelines = inspect.getsourcelines(inner_fun)[0]
        remains = remove_header_parens(sourcelines)
        self.assertEqual(len(remains), 1)
        self.assertEqual(remains[0], '            pass\n')


class TestOrderedSet(unittest.TestCase):
    """ Test cases for OrderedSet """

    def test_ordered_set_creation(self):
        """ Testing OrderedSet creation """
        new_set = OrderedSet([1, 2, 3])
        self.assertIn(1, new_set)
        self.assertIn(2, new_set)
        self.assertIn(3, new_set)

    def test_ordered_set_loop(self):
        """ Testing OrderedSet looping """
        new_set = OrderedSet([1, 2, 3])
        count = 1
        for val in new_set:
            self.assertEqual(count, val)
            count += 1

    def test_ordered_set_eq(self):
        """ Testing OrderedSet equality """
        set_a = OrderedSet([1, 2, 3])
        set_b = OrderedSet([1, 2, 3])
        self.assertEqual(set_a, set_b)

    def test_ordered_set_neq(self):
        """ Testing OrderedSet non-equality """
        set_a = OrderedSet([1, 2, 3])
        set_b = OrderedSet([1, 2, 4])
        self.assertNotEqual(set_a, set_b)

    def test_ordered_set_len(self):
        """ Testing OrderedSet length """
        set_a = OrderedSet([1, 2, 3, 4, 5])
        self.assertEqual(len(set_a), 5)

    def test_ordered_set_disjoint(self):
        """ Testing OrderedSet disjoint """
        set_a = OrderedSet([1, 2, 3, 4])
        set_b = OrderedSet([5, 6, 7, 8])
        self.assertTrue(set_a.isdisjoint(set_b))

    def test_ordered_set_issubset(self):
        """ Testing OrderedSet subsetting """
        set_a = OrderedSet([1, 2, 3])
        set_b = OrderedSet([1, 2, 3, 4, 5])
        self.assertTrue(set_a.issubset(set_b))

    def test_ordered_set_issuperset(self):
        """ Testing OrderedSet supersetting """
        set_a = OrderedSet([1, 2, 3])
        set_b = OrderedSet([1, 2, 3, 4, 5])
        self.assertTrue(set_b.issuperset(set_a))

    def test_ordered_set_union(self):
        """ Testing OrderedSet union """
        set_a = OrderedSet([1, 2, 3])
        set_b = OrderedSet([1, 2, 3, 4, 5])
        union_set = set_a.union(set_b)
        count = 1
        for value in union_set:
            self.assertTrue(count, value)
            count += 1

    def test_ordered_set_intersection(self):
        """ Testing OrderedSet intersection """
        set_a = OrderedSet([1, 2, 3])
        set_b = OrderedSet([1, 2, 3, 4, 5])
        intersect_set = set_a.intersection(set_b)
        count = 1
        for value in intersect_set:
            self.assertTrue(count, value)
            count += 1
        self.assertNotIn(4, intersect_set)
        self.assertNotIn(5, intersect_set)

    def test_ordered_set_diff_empty(self):
        """ Testing OrderedSet difference resulting in an empty set """
        set_a = OrderedSet([1, 2, 3])
        set_b = OrderedSet([1, 2, 3, 4, 5])
        diff_set = set_a - set_b
        self.assertTrue(len(diff_set) == 0)

    def test_ordered_set_difference(self):
        """ Testing OrderedSet difference """
        set_a = OrderedSet([1, 2, 3, 4, 5])
        set_b = OrderedSet([1, 2, 3])
        diff_set = set_a - set_b
        self.assertNotIn(1, diff_set)
        self.assertNotIn(2, diff_set)
        self.assertNotIn(3, diff_set)
        self.assertIn(4, diff_set)
        self.assertIn(5, diff_set)

    def test_symmetric_difference(self):
        """ Testing OrderedSet symmetric difference """
        set_a = OrderedSet([1, 2, 3, 4, 5])
        set_b = OrderedSet([1, 2, 3, 6])
        xor_set = set_a.symmetric_difference(set_b)
        self.assertNotIn(1, xor_set)
        self.assertNotIn(2, xor_set)
        self.assertNotIn(3, xor_set)
        self.assertIn(4, xor_set)
        self.assertIn(5, xor_set)
        self.assertIn(6, xor_set)

if __name__ == "__main__":
    unittest.main()

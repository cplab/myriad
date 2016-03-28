"""
Test cases for myriad_utils.
:author Pedro Rittner
"""

import unittest
import inspect

from collections import OrderedDict

from context import myriad_utils as mutils


class TestRemoveHeaderParens(unittest.TestCase):
    """ Test cases for the function remove_header_parens """

    def test_remove_header_parens_1l(self):
        """ Testing removing a single-line header from a function """
        def inner_fun(a_var: int) -> str:
            pass
        sourcelines = inspect.getsourcelines(inner_fun)[0]
        remains = mutils.remove_header_parens(sourcelines)
        self.assertEqual(len(remains), 1)
        self.assertEqual(remains[0], '            pass\n')

    def test_remove_header_parens_2l(self):
        """ Testing removing a two-line header from a function """
        def inner_fun(a_var: int,
                      b_var: int) -> str:
            pass
        sourcelines = inspect.getsourcelines(inner_fun)[0]
        remains = mutils.remove_header_parens(sourcelines)
        self.assertEqual(len(remains), 1)
        self.assertEqual(remains[0], '            pass\n')


class TestOrderedSet(unittest.TestCase):
    """ Test cases for OrderedSet """

    def test_ordered_set_creation(self):
        """ Testing OrderedSet creation """
        new_set = mutils.OrderedSet([1, 2, 3])
        self.assertIn(1, new_set)
        self.assertIn(2, new_set)
        self.assertIn(3, new_set)

    def test_ordered_set_loop(self):
        """ Testing OrderedSet looping """
        new_set = mutils.OrderedSet([1, 2, 3])
        count = 1
        for val in new_set:
            self.assertEqual(count, val)
            count += 1

    def test_ordered_set_eq(self):
        """ Testing OrderedSet equality """
        set_a = mutils.OrderedSet([1, 2, 3])
        set_b = mutils.OrderedSet([1, 2, 3])
        self.assertEqual(set_a, set_b)

    def test_ordered_set_neq(self):
        """ Testing OrderedSet non-equality """
        set_a = mutils.OrderedSet([1, 2, 3])
        set_b = mutils.OrderedSet([1, 2, 4])
        self.assertNotEqual(set_a, set_b)

    def test_ordered_set_len(self):
        """ Testing OrderedSet length """
        set_a = mutils.OrderedSet([1, 2, 3, 4, 5])
        self.assertEqual(len(set_a), 5)

    def test_ordered_set_disjoint(self):
        """ Testing OrderedSet disjoint """
        set_a = mutils.OrderedSet([1, 2, 3, 4])
        set_b = mutils.OrderedSet([5, 6, 7, 8])
        self.assertTrue(set_a.isdisjoint(set_b))

    def test_ordered_set_issubset(self):
        """ Testing OrderedSet subsetting """
        set_a = mutils.OrderedSet([1, 2, 3])
        set_b = mutils.OrderedSet([1, 2, 3, 4, 5])
        self.assertTrue(set_a.issubset(set_b))

    def test_ordered_set_issuperset(self):
        """ Testing OrderedSet supersetting """
        set_a = mutils.OrderedSet([1, 2, 3])
        set_b = mutils.OrderedSet([1, 2, 3, 4, 5])
        self.assertTrue(set_b.issuperset(set_a))

    def test_ordered_set_union(self):
        """ Testing OrderedSet union """
        set_a = mutils.OrderedSet([1, 2, 3])
        set_b = mutils.OrderedSet([1, 2, 3, 4, 5])
        union_set = set_a.union(set_b)
        count = 1
        for value in union_set:
            self.assertTrue(count, value)
            count += 1

    def test_ordered_set_intersection(self):
        """ Testing OrderedSet intersection """
        set_a = mutils.OrderedSet([1, 2, 3])
        set_b = mutils.OrderedSet([1, 2, 3, 4, 5])
        intersect_set = set_a.intersection(set_b)
        count = 1
        for value in intersect_set:
            self.assertTrue(count, value)
            count += 1
        self.assertNotIn(4, intersect_set)
        self.assertNotIn(5, intersect_set)

    def test_ordered_set_diff_empty(self):
        """ Testing OrderedSet difference resulting in an empty set """
        set_a = mutils.OrderedSet([1, 2, 3])
        set_b = mutils.OrderedSet([1, 2, 3, 4, 5])
        diff_set = set_a - set_b
        self.assertTrue(len(diff_set) == 0)

    def test_ordered_set_difference(self):
        """ Testing OrderedSet difference """
        set_a = mutils.OrderedSet([1, 2, 3, 4, 5])
        set_b = mutils.OrderedSet([1, 2, 3])
        diff_set = set_a - set_b
        self.assertNotIn(1, diff_set)
        self.assertNotIn(2, diff_set)
        self.assertNotIn(3, diff_set)
        self.assertIn(4, diff_set)
        self.assertIn(5, diff_set)

    def test_symmetric_difference(self):
        """ Testing OrderedSet symmetric difference """
        set_a = mutils.OrderedSet([1, 2, 3, 4, 5])
        set_b = mutils.OrderedSet([1, 2, 3, 6])
        xor_set = set_a.symmetric_difference(set_b)
        self.assertNotIn(1, xor_set)
        self.assertNotIn(2, xor_set)
        self.assertNotIn(3, xor_set)
        self.assertIn(4, xor_set)
        self.assertIn(5, xor_set)
        self.assertIn(6, xor_set)


class TestIndentFix(unittest.TestCase):
    """ Tests indent fixer function"""

    def test_empty_str(self):
        """ Testing if indent_fix works on empty strings and None """
        self.assertIsNone(mutils.indent_fix(None))
        self.assertEqual(mutils.indent_fix(""), "")

    def test_single_line(self):
        """ Testing if single-line case works """
        test_str = """     dummy = False     """
        self.assertEqual(mutils.indent_fix(test_str), "dummy = False")

    def test_multi_line(self):
        """ Testing if multi-line case works """
        expected_str = """for i in list(range(1, 10)):
    print(i)
    if i < 10:
        pass
    else:
        print("LOL")"""
        test_str = """
        for i in list(range(1, 10)):
            print(i)
            if i < 10:
                pass
            else:
                print("LOL")"""
        self.assertEqual(expected_str, mutils.indent_fix(test_str))


class TestFilterODictValues(unittest.TestCase):
    """ Tests filtering out values from an OrderedDict"""

    def test_filter_empty_odict(self):
        """ Testing filtering out values from an empty OrderedDict """
        old_d = OrderedDict()
        new_d = mutils.filter_odict_values(old_d)
        self.assertTrue(len(new_d) == len(old_d))

    def test_filter_empty_args(self):
        """ Testing filtering out nothing from a non-empty OrderedDict """
        old_d = OrderedDict({"a": 1, "b": 5.5, "c": "LOL"})
        new_d = mutils.filter_odict_values(old_d)
        self.assertTrue(len(new_d) == len(old_d))

    def test_filter_single_type(self):
        """ Testing filtering out a single type from non-empty OrderedDict """
        old_d = OrderedDict({"a": 1, "b": 5.5, "c": "LOL"})
        new_d = mutils.filter_odict_values(old_d, int)
        self.assertTrue(len(new_d) == 2)

    def test_filter_multiple_types(self):
        """ Testing filtering out multiple types from non-empty OrderedDict """
        old_d = OrderedDict({"a": 1, "b": 5.5, "c": "LOL"})
        new_d = mutils.filter_odict_values(old_d, int, float)
        self.assertTrue(len(new_d) == 1)

    def test_filter_inherited_types(self):
        """ Testing filtering out inherited types """
        class Parent(object):
            pass

        class Child(Parent):
            pass
        old_d = OrderedDict({"a": 1, 45: Parent(), "c": Child()})
        new_d = mutils.filter_odict_values(old_d, Parent)
        self.assertTrue(len(new_d) == 1)
        new_d = mutils.filter_odict_values(old_d, Child)
        self.assertTrue(len(new_d) == 2)

if __name__ == "__main__":
    unittest.main()

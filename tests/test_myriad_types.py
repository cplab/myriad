"""
Test cases for myriad_types.
"""
import unittest

from collections import OrderedDict

from myriad_testing import trim_spaces

from context import myriad
from myriad import myriad_types as mtypes


class TestScalars(unittest.TestCase):
    """
    Test Cases for Scalars.
    """

    def test_qual_scalar(self):
        """ Testing qualified scalar creation """
        void_ptr = mtypes.MyriadScalar(
            "self", mtypes.MVoid, True, quals=["const"])
        self.assertEqual("const void *self", void_ptr.stringify_decl())


class TestFunctions(unittest.TestCase):
    """
    Test Cases for Functions
    """

    def test_void_ret_function(self):
        """ Testing for functions with void return types """
        void_ptr = mtypes.MyriadScalar(
            "self", mtypes.MVoid, True, quals=["const"])
        myriad_dtor = mtypes.MyriadFunction(
            "dtor", OrderedDict({0: void_ptr}))
        self.assertEqual("void dtor(const void *self)",
                         myriad_dtor.stringify_decl())

    def test_function_typedef(self):
        """ Testing function typedef generation """
        void_ptr = mtypes.MyriadScalar(
            "self", mtypes.MVoid, True, quals=["const"])
        dtor = mtypes.MyriadFunction("dtor", OrderedDict({0: void_ptr}))
        dtor.gen_typedef()
        self.assertEqual("typedef void (*dtor_t)(const void *self)",
                         dtor.stringify_typedef())


class TestStructs(unittest.TestCase):
    """
    Test Cases for Structs
    """

    def test_single_member_struct(self):
        """ Testing having a struct with single member """
        void_ptr = mtypes.MyriadScalar(
            "self", mtypes.MVoid, True, quals=["const"])
        myriad_class = mtypes.MyriadStructType(
            "MyriadClass", OrderedDict({0: void_ptr}))
        str1 = myriad_class.stringify_decl().replace('\n', ' ')
        str2 = "struct MyriadClass {   const void *self; }"
        self.assertEqual(trim_spaces(str1), trim_spaces(str2))

    def test_struct_ptr(self):
        """ Testing having a struct pointer variable """
        void_ptr = mtypes.MyriadScalar(
            "self", mtypes.MVoid, True, quals=["const"])
        myriad_class = mtypes.MyriadStructType(
            "MyriadClass", OrderedDict({0: void_ptr}))
        class_2 = myriad_class("class_2", ptr=True)
        self.assertEqual(trim_spaces("struct MyriadClass *class_2"),
                         trim_spaces(class_2.stringify_decl()))

    def test_struct_qual(self):
        """ Testing making a struct having a qualifier """
        void_ptr = mtypes.MyriadScalar(
            "self", mtypes.MVoid, True, quals=["const"])
        myriad_class = mtypes.MyriadStructType(
            "MyriadClass", OrderedDict({0: void_ptr}))
        class_m = myriad_class("class_m", quals=["const"])
        self.assertEqual("const struct MyriadClass class_m",
                         class_m.stringify_decl())

    def test_struct_nesting(self):
        """ Testing basic struct nesting """
        void_ptr = mtypes.MyriadScalar(
            "self", mtypes.MVoid, True, quals=["const"])
        double_val = mtypes.MyriadScalar("val", mtypes.MDouble)
        myriad_class = mtypes.MyriadStructType(
            "MyriadClass", OrderedDict({0: void_ptr}))
        class_m = myriad_class("class_m", quals=["const"])
        toplevel_struct = mtypes.MyriadStructType(
            "toplevel", OrderedDict({0: class_m, 1: double_val}))
        expected_result = \
            "struct toplevel { const struct MyriadClass class_m; double val; }"
        str_result = toplevel_struct.stringify_decl().replace('\n', ' ')
        self.assertEqual(trim_spaces(expected_result),
                         trim_spaces(str_result))


class TestArray(unittest.TestCase):
    """
    Test Cases for Arrays
    """

    def test_array_full(self):
        """ Testing arrays with qualifiers, storage specifies, and dim IDs """
        my_arr = mtypes.MyriadScalar(
            ident="my_arr",
            base_type=mtypes.MDouble,
            ptr=False,
            quals=["const"],
            storage=["static"],
            arr_id="SIMUL_LEN")
        self.assertEqual("static const double my_arr[SIMUL_LEN]",
                         my_arr.stringify_decl())


class TestCasts(unittest.TestCase):
    """
    Test Cases for Casting
    """

    def test_cast_to_parent(self):
        """ Testing for a simple cast where the field is in the struct """
        void_ptr = mtypes.MyriadScalar(
            "self", mtypes.MVoid, True, quals=["const"])
        double_val = mtypes.MyriadScalar("val", mtypes.MDouble)
        myriad_class = mtypes.MyriadStructType(
            "MyriadClass", OrderedDict({0: void_ptr, 1: double_val}))
        self.assertEqual("(struct MyriadClass*)",
                         mtypes.cast_to_parent(myriad_class, "val"))

    def test_cast_to_parent_nested(self):
        """ Testing for a cast with 1 level of recursion """
        void_ptr = mtypes.MyriadScalar(
            "self", mtypes.MVoid, True, quals=["const"])
        double_val = mtypes.MyriadScalar("val", mtypes.MDouble)
        myriad_class = mtypes.MyriadStructType(
            "MyriadClass", OrderedDict({0: void_ptr}))
        class_m = myriad_class("_", quals=["const"])
        toplevel_struct = mtypes.MyriadStructType(
            "toplevel", OrderedDict({0: class_m, 1: double_val}))
        self.assertEqual("(struct MyriadClass*)",
                         mtypes.cast_to_parent(toplevel_struct, "self"))


if __name__ == '__main__':
    unittest.main()

"""
Test cases for myriad_types.
"""
import unittest

from collections import OrderedDict

from myriad_types import MVoid, MDouble
from myriad_types import MyriadScalar, MyriadFunction, MyriadStructType
from myriad_types import cast_to_parent


class TestScalars(unittest.TestCase):
    """
    Test Cases for Scalars.
    """

    def test_qual_scalar(self):
        """ Testing qualified scalar creation """
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        self.assertEqual("const void *self", void_ptr.stringify_decl())


class TestFunctions(unittest.TestCase):
    """
    Test Cases for Functions
    """

    def test_void_ret_function(self):
        """ Testing for functions with void return types """
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        myriad_dtor = MyriadFunction("dtor", OrderedDict({0: void_ptr}))
        self.assertEqual("void dtor(const void *self)",
                         myriad_dtor.stringify_decl())

    def test_function_typedef(self):
        """ Testing function typedef generation """
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        dtor = MyriadFunction("dtor", OrderedDict({0: void_ptr}))
        dtor.gen_typedef()
        self.assertEqual("typedef void (*dtor_t)(const void *self)",
                         dtor.stringify_typedef())


class TestStructs(unittest.TestCase):
    """
    Test Cases for Structs
    """

    def test_single_member_struct(self):
        """ Testing having a struct with single member """
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        myriad_class = MyriadStructType("MyriadClass",
                                        OrderedDict({0: void_ptr}))
        str1 = myriad_class.stringify_decl().replace('\n', ' ')
        str2 = "struct MyriadClass {   const void *self; }"
        self.assertEqual(str1, str2)

    def test_struct_ptr(self):
        """ Testing having a struct pointer variable """
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        myriad_class = MyriadStructType("MyriadClass",
                                        OrderedDict({0: void_ptr}))
        class_2 = myriad_class("class_2", ptr=True)
        self.assertEqual("struct MyriadClass *class_2",
                         class_2.stringify_decl())

    def test_struct_qual(self):
        """ Testing making a struct having a qualifier """
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        myriad_class = MyriadStructType("MyriadClass",
                                        OrderedDict({0: void_ptr}))
        class_m = myriad_class("class_m", quals=["const"])
        self.assertEqual("const struct MyriadClass class_m",
                         class_m.stringify_decl())

    def test_struct_nesting(self):
        """ Testing basic struct nesting """
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        double_val = MyriadScalar("val", MDouble)
        myriad_class = MyriadStructType("MyriadClass",
                                        OrderedDict({0: void_ptr}))
        class_m = myriad_class("class_m", quals=["const"])
        toplevel_struct = MyriadStructType("toplevel",
                                           OrderedDict({0: class_m,
                                                        1: double_val}))
        self.assertEqual(
            "struct toplevel {   const struct MyriadClass class_m;   double val; }",
            toplevel_struct.stringify_decl().replace('\n', ' '))


class TestArray(unittest.TestCase):
    """
    Test Cases for Arrays
    """

    def test_array_full(self):
        """ Testing arrays with qualifiers, storage specifies, and dim IDs """
        my_arr = MyriadScalar(ident="my_arr",
                              base_type=MDouble,
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
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        double_val = MyriadScalar("val", MDouble)
        myriad_class = MyriadStructType("MyriadClass",
                                        OrderedDict({0: void_ptr,
                                                     1: double_val}))
        self.assertEqual("(struct MyriadClass*)",
                         cast_to_parent(myriad_class, "val"))

    def test_cast_to_parent_nested(self):
        """ Testing for a cast with 1 level of recursion """
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        double_val = MyriadScalar("val", MDouble)
        myriad_class = MyriadStructType("MyriadClass",
                                        OrderedDict({0: void_ptr}))
        class_m = myriad_class("_", quals=["const"])
        toplevel_struct = MyriadStructType("toplevel",
                                           OrderedDict({0: class_m,
                                                        1: double_val}))
        self.assertEqual("(struct MyriadClass*)",
                         cast_to_parent(toplevel_struct, "self"))


if __name__ == '__main__':
    unittest.main()

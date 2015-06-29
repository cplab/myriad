"""
Test cases for myriad_types.
"""
import unittest

from collections import OrderedDict

from myriad_types import MVoid, MDouble
from myriad_types import MyriadScalar, MyriadFunction, MyriadStructType


class TestScalars(unittest.TestCase):
    """
    Test Cases for Scalars.
    """

    def test_void_ptr_scalar(self):
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        self.assertEqual("const void *self", void_ptr.stringify_decl())


class TestFunctions(unittest.TestCase):
    """
    Test Cases for Functions
    """

    def test_void_ptr_function(self):
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        myriad_dtor = MyriadFunction("dtor", OrderedDict({0: void_ptr}))
        self.assertEqual("void dtor(const void *self)",
                         myriad_dtor.stringify_decl())

    def test_void_ptr_function_typedef(self):
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        dtor = MyriadFunction("dtor", OrderedDict({0: void_ptr}))
        dtor.gen_typedef()
        self.assertEqual("typedef void (*dtor_t)(const void *self)",
                         dtor.stringify_typedef())


class TestStructs(unittest.TestCase):
    """
    Test Cases for Structs
    """

    def test_struct(self):
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        myriad_class = MyriadStructType("MyriadClass",
                                        OrderedDict({0: void_ptr}))
        str1 = myriad_class.stringify_decl().replace('\n', ' ')
        str2 = "struct MyriadClass {   const void *self; }"
        self.assertEqual(str1, str2)

    def test_struct_ptr(self):
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        myriad_class = MyriadStructType("MyriadClass",
                                        OrderedDict({0: void_ptr}))
        class_2 = myriad_class("class_2", ptr=True)
        self.assertEqual("struct MyriadClass *class_2",
                         class_2.stringify_decl())

    def test_struct_const(self):
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        myriad_class = MyriadStructType("MyriadClass",
                                        OrderedDict({0: void_ptr}))
        class_m = myriad_class("class_m", quals=["const"])
        self.assertEqual("const struct MyriadClass class_m",
                         class_m.stringify_decl())


class TestArray(unittest.TestCase):
    """
    Test Cases for Arrays
    """

    def test_array(self):
        my_arr = MyriadScalar(ident="my_arr",
                              base_type=MDouble,
                              ptr=False,
                              quals=["const"],
                              storage=["static"],
                              arr_id="SIMUL_LEN")
        self.assertEqual("static const double my_arr[SIMUL_LEN]",
                         my_arr.stringify_decl())


if __name__ == '__main__':
    unittest.main()

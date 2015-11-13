"""
..module:: test_myriad_metaclass
    :platform: Linux
    :synopsis: Testing for metaclass integration

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>
"""

import unittest
import os

from collections import OrderedDict

from myriad_testing import set_external_loggers, MyriadTestCase

from myriad_types import MyriadFunction, MyriadScalar, MVoid, MInt, MDouble
from myriad_types import MyriadTimeseriesVector

import myriad_metaclass


@set_external_loggers("TestMyriadMethod", myriad_metaclass.LOG)
class TestMyriadMethod(MyriadTestCase):
    """
    Tests Myriad Method functionality 'standalone'
    """

    def test_delg_template_init(self):
        """ Testing if delegator template is correctly initialized """
        with open("templates/delegator_func.mako", 'r') as delegator_f:
            contents = delegator_f.read()
            self.assertEqual(contents, myriad_metaclass.DELG_TEMPLATE)

    def test_super_delg_template_init(self):
        """ Testing if super delegator template is correctly initialized """
        with open("templates/super_delegator_func.mako", 'r') as s_delg_f:
            contents = s_delg_f.read()
            self.assertEqual(contents, myriad_metaclass.SUPER_DELG_TEMPLATE)

    def test_create_super_delegator(self):
        """ Testing if creating super delegator functions works """
        void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
        myriad_dtor = MyriadFunction("dtor", OrderedDict({0: void_ptr}))
        classname = "Compartment"
        super_delg = myriad_metaclass.create_super_delegator(myriad_dtor,
                                                             classname)
        # Compare result strings
        expected_result = """
        void super_dtor(const void *_class, const void *self)
        {
        const struct Compartment* superclass = (const struct Compartment*)
            myriad_super(_class);
        return superclass->my_dtor_t(self);
        }
        """
        self.assertTrimStrEquals(str(super_delg), expected_result)

    def test_create_delegator(self):
        """ Testing if creating delegators works """
        # Create scalars and function
        args_list = OrderedDict()
        args_list["self"] = MyriadScalar("_self", MVoid, ptr=True)
        args_list["mechanism"] = MyriadScalar("mechanism", MVoid, ptr=True)
        instance_fxn = MyriadFunction("add_mech",
                                      args_list,
                                      MyriadScalar("_", MInt),
                                      ["static"])
        # Generate delegator
        classname = "Compartment"
        result_fxn = myriad_metaclass.create_delegator(instance_fxn, classname)
        # Compare result strings
        expected_result = """
        int64_t add_mech(void *_self, void *mechanism)
        {
        const struct Compartment* m_class = (const struct Compartment*)
            myriad_class_of(_self);
        assert(m_class->my_add_mech_t);
        return m_class->my_add_mech_t(mechanism);
        }
        """
        self.assertTrimStrEquals(str(result_fxn), expected_result)


@set_external_loggers("TestMyriadMetaclass", myriad_metaclass.LOG)
class TestMyriadMetaclass(MyriadTestCase):
    """
    Tests MyriadMetaclass functionality 'standalone'
    """

    # Current error/failure count
    curr_errors = 0
    curr_failures = 0

    def test_create_blank_class(self):
        """ Testing if creating a blank metaclass works """
        class BlankObj(myriad_metaclass.MyriadObject):
            """ Blank test class """
            pass
        # Test for the existence expected members
        self.assertTrue("obj_struct" in BlankObj.__dict__)
        self.assertTrue("myriad_obj_vars" in BlankObj.__dict__)
        # TODO: Add more checks for other members
        # Check if names are as we expect them
        self.assertTrue("obj_name" in BlankObj.__dict__)
        self.assertEqual(BlankObj.__dict__["obj_name"], "BlankObj")
        self.assertTrue("cls_name" in BlankObj.__dict__)
        self.assertEqual(BlankObj.__dict__["cls_name"], "BlankObjClass")

    def test_create_variable_only_class(self):
        """ Testing if creating a variable-only metaclass works """
        class VarOnlyObj(myriad_metaclass.MyriadObject):
            capacitance = MDouble
            vm = MyriadScalar("vm", MDouble, ptr=True)
        result_str = VarOnlyObj.__dict__["obj_struct"].stringify_decl()
        expected_result = """
        struct VarOnlyObj
        {
            const struct MyriadObject _;
            double capacitance;
            double *vm;
        }
        """
        self.assertTrimStrEquals(result_str, expected_result)

    def test_create_methods_class(self):
        """ Testing if creating Myriad classes with methods works """
        class MethodsObj(myriad_metaclass.MyriadObject):
            @myriad_metaclass.myriad_method
            def do_stuff(self):
                return 0
        # Test whether a function pointer scalar is created
        self.assertIn("myriad_cls_vars", MethodsObj.__dict__)
        self.assertIn("my_do_stuff_t", MethodsObj.__dict__["myriad_cls_vars"])
        # Test whether a myriad method was created
        self.assertIn("myriad_methods", MethodsObj.__dict__)
        self.assertIn("do_stuff", MethodsObj.__dict__["myriad_methods"])

    def test_verbatim_methods(self):
        """ Testing if creating Myriad classes with verbatim methods work"""
        class VerbatimObj(myriad_metaclass.MyriadObject):
            @myriad_metaclass.myriad_method_verbatim
            def do_verbatim_stuff(self):
                """return;"""
        self.assertIn("myriad_cls_vars", VerbatimObj.__dict__)
        self.assertIn("my_do_verbatim_stuff_t",
                      VerbatimObj.__dict__["myriad_cls_vars"])
        self.assertIsNotNone(
            VerbatimObj.__dict__["myriad_methods"]["do_verbatim_stuff"])


@set_external_loggers("TestMyriadRendering", myriad_metaclass.LOG)
class TestMyriadRendering(MyriadTestCase):
    """
    Tests MyriadMetaclass rendering of objects
    """

    def assertFilesExist(self, base_cls):
        """ Raises AssertionError if template files do not exist """
        base_name = base_cls.__name__
        file_list = [base_name + ".c",
                     base_name + ".h",
                     base_name + ".cuh",
                     "py_" + base_name + ".c"]
        for filename in file_list:
            if not os.path.isfile(filename):
                raise AssertionError("Template file not found: " + filename)

    def test_template_instantiation(self):
        """ Testing if template rendering produces files """
        class RenderObj(myriad_metaclass.MyriadObject):
            pass
        RenderObj.render_templates()
        self.assertFilesExist(RenderObj)

    def test_render_variable_only_class(self):
        """ Testing if rendering a variable-only class works """
        class VarOnlyObj(myriad_metaclass.MyriadObject):
            capacitance = MDouble
        VarOnlyObj.render_templates()
        self.assertFilesExist(VarOnlyObj)

    def test_render_timeseries_class(self):
        """ Testing if rendering a timeseries-containing class works"""
        class TimeseriesObj(myriad_metaclass.MyriadObject):
            capacitance = MDouble
            vm = MyriadTimeseriesVector
        TimeseriesObj.render_templates()
        self.assertFilesExist(TimeseriesObj)


if __name__ == '__main__':
    unittest.main()

"""
..module:: test_myriad_object
    :platform: Linux
    :synopsis: Testing for inheriting from MyriadObject

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>
"""

import unittest

from myriad_testing import set_external_loggers, MyriadTestCase

from myriad_types import MInt, MDouble
from myriad_types import MyriadTimeseriesVector

from myriad_object import LOG as MYRIAD_OBJECT_LOG
from myriad_object import MyriadObject, myriad_method, myriad_method_verbatim


@set_external_loggers("TestMyriadMetaclass", MYRIAD_OBJECT_LOG)
class TestMyriadMetaclass(MyriadTestCase):
    """
    Tests MyriadMetaclass functionality 'standalone'
    """

    # Current error/failure count
    curr_errors = 0
    curr_failures = 0

    def test_create_blank_class(self):
        """ Testing if creating a blank metaclass works """
        class BlankObj(MyriadObject):
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
        class VarOnlyObj(MyriadObject):
            capacitance = MDouble
            vm = MyriadTimeseriesVector
        result_str = VarOnlyObj.__dict__["obj_struct"].stringify_decl()
        expected_result = """
        struct VarOnlyObj
        {
            const struct MyriadObject _;
            double capacitance;
            double vm[SIMUL_LEN];
        }
        """
        self.assertTrimStrEquals(result_str, expected_result)

    def test_create_methods_class(self):
        """ Testing if creating Myriad classes with methods works """
        class MethodsObj(MyriadObject):
            @myriad_method
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
        class VerbatimObj(MyriadObject):
            @myriad_method_verbatim
            def do_verbatim_stuff(self):
                """return;"""
        self.assertIn("myriad_cls_vars", VerbatimObj.__dict__)
        self.assertIn("my_do_verbatim_stuff_t",
                      VerbatimObj.__dict__["myriad_cls_vars"])
        self.assertIsNotNone(
            VerbatimObj.__dict__["myriad_methods"]["do_verbatim_stuff"])


@set_external_loggers("TestMyriadRendering", MYRIAD_OBJECT_LOG)
class TestMyriadRendering(MyriadTestCase):
    """
    Tests rendering of objects inheriting from MyriadObject
    """

    def test_template_instantiation(self):
        """ Testing if template rendering produces files """
        class RenderObj(MyriadObject):
            pass
        RenderObj.render_templates()
        self.assertFilesExist(RenderObj)
        self.cleanupFiles(RenderObj)

    def test_render_variable_only_class(self):
        """ Testing if rendering a variable-only class works """
        class VarOnlyObj(MyriadObject):
            capacitance = MDouble
        VarOnlyObj.render_templates()
        self.assertFilesExist(VarOnlyObj)
        self.cleanupFiles(VarOnlyObj)

    def test_render_timeseries_class(self):
        """ Testing if rendering a timeseries-containing class works"""
        class TimeseriesObj(MyriadObject):
            capacitance = MDouble
            vm = MyriadTimeseriesVector
        TimeseriesObj.render_templates()
        self.assertFilesExist(TimeseriesObj)
        self.cleanupFiles(TimeseriesObj)


if __name__ == '__main__':
    unittest.main()

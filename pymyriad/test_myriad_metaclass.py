"""
..module:: test_myriad_metaclass
    :platform: Linux
    :synopsis: Testing for metaclass integration

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>
"""

import unittest
import logging
import io
import sys
import os

from collections import OrderedDict

from myriad_types import MyriadFunction, MyriadScalar, MVoid, MInt, MDouble
import myriad_metaclass

# Log file
LOG_FILE = io.StringIO()
LOG_LINES = []


def setUpModule():
    """ Logging setup """
    # Create logger with level
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    # Create log handler
    log_handler = logging.StreamHandler(stream=LOG_FILE)
    log_handler.setLevel(logging.DEBUG)
    # Create and set log formatter
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_handler.setFormatter(log_formatter)
    # Set log handler
    log.addHandler(log_handler)
    # Add handler/formatter to module we're testing
    myriad_metaclass.LOG.addHandler(log_handler)
    myriad_metaclass.LOG.setLevel(logging.DEBUG)


def tearDownModule():
    print("", file=sys.stderr)
    for line in LOG_LINES:
        print(line, file=sys.stderr)


class TestMyriadMethod(unittest.TestCase):
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
        expected_result = " ".join("""
        void super_dtor(const void *_class, const void *self)
        {
        const struct Compartment* superclass = (const struct Compartment*)
            myriad_super(_class);
        return superclass->my_dtor_t(self);
        }
        """.split())
        result_str = " ".join(str(super_delg).split())
        self.assertEqual(result_str, expected_result)

    def test_create_delegator(self):
        """ Testing if creating delegators works """
        # Create scalars and function
        args_list = OrderedDict()
        args_list["self"] = MyriadScalar("_self", MVoid, ptr=True)
        args_list["mechanism"] = MyriadScalar("mechanism", MVoid, ptr=True)
        instance_fxn = MyriadFunction("Compartment_add_mech",
                                      args_list,
                                      MyriadScalar("_", MInt),
                                      ["static"])
        # Generate delegator
        classname = "Compartment"
        result_fxn = myriad_metaclass.create_delegator(instance_fxn, classname)
        # Compare result strings
        expected_result = " ".join("""
        int64_t add_mech(void *_self, void *mechanism)
        {
        const struct Compartment* m_class = (const struct Compartment*)
            myriad_class_of(_self);
        assert(m_class->my_add_mech_t);
        return m_class->my_add_mech_t(mechanism);
        }
        """.split())
        result_str = " ".join(str(result_fxn).split())
        self.assertEqual(result_str, expected_result)


class TestMyriadMetaclass(unittest.TestCase):
    """
    Tests MyriadMetaclass functionality 'standalone'
    """

    # Current error/failure count
    curr_errors = 0
    curr_failures = 0

    def run(self, result=None):
        """ Show log output on failed tests """
        curr_errors = len(result.errors) if result else 0
        curr_failures = len(result.failures) if result else 0
        if self.curr_errors < curr_errors:
            self.curr_errors = curr_errors
            LOG_LINES.append("Error @ " + result.errors[0][0].__repr__() +
                             ":\n" + LOG_FILE.getvalue())
        elif self.curr_failures < curr_failures:
            self.curr_failures = curr_failures
            LOG_LINES.append("Failure @ " + result.failures[0][0].__repr__() +
                             ":\n" + LOG_FILE.getvalue())
        else:
            # Truncate the StringIO
            LOG_FILE.truncate(0)
        return super().run(result)

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
        result_str = " ".join(
            VarOnlyObj.__dict__["obj_struct"].stringify_decl().split())
        expected_result = " ".join("""
        struct VarOnlyObj
        {
            const struct MyriadObject _;
            double capacitance;
            double *vm;
        }
        """.split())
        self.assertEqual(result_str, expected_result)

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

    def test_template_rendering(self):
        """ Testing if template rendering works """
        class RenderObj(myriad_metaclass.MyriadObject):
            pass
        RenderObj.render_templates()
        self.assertTrue(os.path.isfile("RenderObj.c"))


def main():
    """ Runs the tests, doing some setup. """
    unittest.main(buffer=False)


if __name__ == '__main__':
    main()

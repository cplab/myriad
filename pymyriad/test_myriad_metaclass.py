"""
..module:: test_myriad_metaclass
    :platform: Linux
    :synopsis: Testing for metaclass integration

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>
"""

import unittest

from collections import OrderedDict

from myriad_testing import set_external_loggers, MyriadTestCase

from myriad_types import MyriadFunction, MyriadScalar, MVoid, MInt

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

if __name__ == '__main__':
    unittest.main()

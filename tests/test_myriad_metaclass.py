"""
..module:: test_myriad_metaclass
    :platform: Linux
    :synopsis: Testing for metaclass integration

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>
"""

import unittest

from collections import OrderedDict

from myriad_testing import set_external_loggers, MyriadTestCase

from context import myriad
from myriad import myriad_types
from myriad import myriad_metaclass


@set_external_loggers("TestMyriadMethod", myriad_metaclass.LOG)
class TestMyriadMethod(MyriadTestCase):
    """
    Tests Myriad Method functionality 'standalone'
    """

    def test_create_super_delegator(self):
        """ Testing if creating super delegator functions works """
        void_ptr = myriad_types.MyriadScalar(
            "self", myriad_types.MVoid, True, quals=["const"])
        myriad_dtor = myriad_types.MyriadFunction(
            "dtor", OrderedDict({0: void_ptr}))
        classname = "Compartment"
        super_delg = myriad_metaclass.create_super_delegator(myriad_dtor,
                                                             classname)
        # Compare result strings
        expected_result = """
        void super_dtor(const int_fast32_t _class, const void *self)
        {
        dtor_vtable[_class](self);
        return;
        }
        """
        self.assertTrimStrEquals(str(super_delg), expected_result)

    def test_create_delegator(self):
        """ Testing if creating delegators works """
        # Create scalars and function
        args_list = OrderedDict()
        args_list["self"] = myriad_types.MyriadScalar(
            "_self", myriad_types.MVoid, ptr=True)
        args_list["mechanism"] = myriad_types.MyriadScalar(
            "mechanism", myriad_types.MVoid, ptr=True)
        instance_fxn = myriad_types.MyriadFunction(
            "add_mech",
            args_list,
            myriad_types.MyriadScalar("_", myriad_types.MInt),
            ["static"])
        # Generate delegator
        classname = "Compartment"
        result_fxn = myriad_metaclass.create_delegator(instance_fxn, classname)
        # Compare result strings
        expected_result = """
        int_fast32_t add_mech(void *_self, void *mechanism)
        {
        return add_mech_vtable[((struct MyriadObject*) self)->class_id](self, mechanism);
        }
        """
        self.assertTrimStrEquals(str(result_fxn), expected_result)

if __name__ == '__main__':
    unittest.main()

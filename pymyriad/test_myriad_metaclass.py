"""
..module:: test_myriad_metaclass
    :platform: Linux
    :synopsis: Testing for metaclass integration

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>
"""

import unittest

from collections import OrderedDict

from myriad_types import MyriadFunction, MyriadScalar, MVoid, MInt
import myriad_metaclass


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

    # def test_create_super_delegator(self):
    #     """ Testing if creating super delegator functions works """
    #     void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
    #     myriad_dtor = MyriadFunction("dtor", OrderedDict({0: void_ptr}))
    #     super_delg = myriad_metaclass.create_super_delegator(myriad_dtor)
    #     self.assertTrue(super_delg.ident.startswith("super_"))

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


def main():
    """ Runs the tests, doing some setup. """
    unittest.main()

if __name__ == '__main__':
    main()

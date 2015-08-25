"""
..module:: test_myriad_metaclass
    :platform: Linux
    :synopsis: Testing for metaclass integration

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>
"""

import unittest

from collections import OrderedDict

from myriad_types import MyriadFunction, MDouble, MyriadScalar
from myriad_metaclass import MyriadMethod


class TestMyriadMethod(unittest.TestCase):
    """
    Tests Myriad Method functionality 'standalone'
    """

    def test_delg_template_init(self):
        """ Testing if delegator template is correctly initialized """
        with open("templates/delegator_func.mako", 'r') as delegator_f:
            contents = delegator_f.read()
            self.assertEqual(contents, MyriadMethod.DELG_TEMPLATE)

    def test_super_delg_template_init(self):
        """ Testing if super delegator template is correctly initialized """
        with open("templates/super_delegator_func.mako", 'r') as s_delg_f:
            contents = s_delg_f.read()
            self.assertEqual(contents, MyriadMethod.SUPER_DELG_TEMPLATE)

    def test_method_init(self):
        mfun_args = OrderedDict({"arg1": MyriadScalar("arg1", MDouble),
                                 "arg2": MyriadScalar("arg1", MDouble)})
        mfun = MyriadFunction("test_fun", mfun_args)
        method = MyriadMethod(mfun, {"TestObject": "return;"})
        self.assertFalse(method.inherited)


def main():
    """ Runs the tests, doing some setup. """
    unittest.main()

if __name__ == '__main__':
    main()

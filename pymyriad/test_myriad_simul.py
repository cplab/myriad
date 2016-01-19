"""
Tests myriad simulation objects
"""

import unittest

from myriad_testing import set_external_loggers, MyriadTestCase

import myriad_simul

import myriad_object


@set_external_loggers("TestMyriadSimulObject", myriad_simul.LOG)
class TestMyriadMethod(MyriadTestCase):

    def test_simul_obj_init(self):
        """ Tests if creating a simulation object works """
        class TestClass(myriad_simul.MyriadSimul,
                        dependencies=["MyriadObject"]):
            pass


class DSACSimul(myriad_simul.MyriadSimul,
                dependencies=[myriad_simul.MyriadObject]):
    pass


def main():
    unittest.main()

if __name__ == '__main__':
    main()

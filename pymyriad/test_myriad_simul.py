"""
Tests myriad simulation objects
"""

import unittest

from myriad_testing import set_external_loggers, MyriadTestCase

import myriad_simul

import myriad_object

import myriad_compartment


@set_external_loggers("TestMyriadSimulObject", myriad_simul.LOG)
class TestMyriadSimulObject(MyriadTestCase):

    def test_simul_obj_init(self):
        """ Tests if creating a simulation object works """
        class TestClass(myriad_simul.MyriadSimul,
                        dependencies=["MyriadObject"]):
            pass
        obj = TestClass()
        self.assertIsNotNone(obj)

    def test_simul_obj_setup(self):
        """ Tests if setting up a simulation object behaves as expected """
        class TestSimul(myriad_simul.MyriadSimul,
                        dependencies=[myriad_object.MyriadObject]):
            pass
        obj = TestSimul()
        self.assertRaises(NotImplementedError, obj.setup)  # Should fail

        class TestSimulSetup(myriad_simul.MyriadSimul,
                             dependencies=[myriad_object.MyriadObject]):
            def setup(self):
                self.compartments.append(myriad_compartment.Compartment())
        obj = TestSimulSetup()
        obj.setup()


def main():
    unittest.main()

if __name__ == '__main__':
    main()

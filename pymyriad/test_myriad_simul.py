"""
Tests myriad simulation objects
"""

import unittest

from myriad_testing import set_external_loggers, MyriadTestCase

import myriad_simul

import myriad_object
import myriad_compartment
import myriad_mechanism


@set_external_loggers("TestMyriadSimulObject", myriad_simul.LOG)
class TestMyriadSimulObject(MyriadTestCase):

    # def test_simul_obj_init(self):
    #     """ Tests if creating a simulation object works """
    #     class TestClass(myriad_simul.MyriadSimul,
    #                     dependencies=[myriad_object.MyriadObject]):
    #         pass
    #     obj = TestClass()
    #     self.assertIsNotNone(obj)

    # def test_simul_obj_setup(self):
    #     """ Tests if setting up a simulation object behaves as expected """
    #     class TestSimul(myriad_simul.MyriadSimul,
    #                     dependencies=[myriad_object.MyriadObject]):
    #         pass
    #     obj = TestSimul()
    #     self.assertRaises(NotImplementedError, obj.setup)  # Should fail

    #     class TestSimulSetup(myriad_simul.MyriadSimul,
    #                          dependencies=[myriad_object.MyriadObject]):
    #         def setup(self):
    #             comp = myriad_compartment.Compartment(cid=0, num_mechs=0)
    #             self.add_compartment(comp)
    #     obj = TestSimulSetup()
    #     obj.setup()

    def test_simul_run(self):
        """ Tests running a simulation object """
        class TestSimul(myriad_simul.MyriadSimul,
                        dependencies=[myriad_object.MyriadObject,
                                      myriad_compartment.Compartment,
                                      myriad_mechanism.Mechanism]):
            def setup(self):
                comp = myriad_compartment.Compartment(cid=0, num_mechs=0)
                self.add_compartment(comp)
        obj = TestSimul(DEBUG=True)
        obj.setup()
        comm = obj.run()
        new_obj = comm.request_data(0)
        print(new_obj)
        comm.close_connection()


def main():
    unittest.main()

if __name__ == '__main__':
    main()

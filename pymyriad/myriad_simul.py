"""
Myriad Simulation Object
"""

import logging

#######
# Log #
#######

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class _MyriadSimulParent(object):
    """ Empty object used for type-checking. """
    pass


class _MyriadSimulMeta(type):
    """
    Metaclass used to automatically parse simulation objects.
    """

    def __new__(mcs, name, bases, namespace, **kwds):
        if len(bases) > 1:
            raise NotImplementedError("Multiple inheritance is not supported.")
        supercls = bases[0]
        if supercls is not _MyriadSimulParent:
            for module in kwds['dependencies']:
                LOG.debug(str(module) + "module imported")
        return super().__new__(mcs, name, bases, namespace)

    @classmethod
    def __prepare__(mcs, name, bases, **kwds):
        return super().__prepare__(name, bases, **kwds)

    def __init__(cls, name, bases, namespace, **kwds):
        super().__init__(name, bases, namespace)


class MyriadSimul(_MyriadSimulParent, metaclass=_MyriadSimulMeta):
    """
    Myriad Simulation object, holding object state and communicating with
    simulation process, including saving data.
    """

    def __init__(self, compartments: list=None, mechanisms: list=None):
        self.compartments = compartments if compartments else []
        self.mechanisms = mechanisms if mechanisms else []

    def add_mechanism(self, comp, mech):
        """ 'Adds' the mechanism to the compartment """
        if comp is None:
            raise ValueError("Cannot add a mechanism to a null Compartment")
        elif mech is None:
            raise ValueError("Cannot add a null mechanism to a Compartment")
        # TODO: Do additional overhead work to 'link' mechanism to compartment
        self.mechanisms.append(mech)

    def setup(self):
        """ Creates and links Compartments and mechanisms """
        raise NotImplementedError("Please override setup() in your class")

    def run(self):
        """ Runs the simulation and puts results back into Python objects """
        raise NotImplementedError("IMPLEMENT ME!")

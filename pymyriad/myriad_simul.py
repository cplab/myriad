"""
Myriad Simulation Object
"""

import os
import sys
import logging
import subprocess

from pkg_resources import resource_string

from myriad_mako_wrapper import MakoFileTemplate

#############
# Templates #
#############

#: Template for makefile (used for building C backend)
MAKEFILE_TEMPLATE = resource_string(__name__,
                                    "templates/Makefile.mako").decode("UTF-8")

#: Template for setup.py (used for building CPython extension modules)
SETUPPY_TEMPLATE = resource_string(__name__,
                                   "templates/setup.py.mako").decode("UTF-8")

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
        dependencies = set()
        if supercls is not _MyriadSimulParent:
            for module in kwds['dependencies']:
                dependencies.add(module)
                LOG.debug(str(module) + "module imported")
        namespace["dependencies"] = dependencies
        return super().__new__(mcs, name, bases, namespace)

    @classmethod
    def __prepare__(mcs, name, bases, **kwds):
        return super().__prepare__(name, bases, **kwds)

    def __init__(cls, name, bases, namespace, **kwds):
        super().__init__(name, bases, namespace)


def _setup_simul_params(params, dependencies) -> dict:
    """ Intelligently discovers and sets parameters """
    # TODO: Intelligently query for nvcc/etc architectures & versions
    if "CUDA_PATH" not in params:
        params["CUDA_PATH"] = os.getenv("CUDA_PATH", "")
    if "CC" not in params:
        default_cc = "gcc.exe" if sys.platform.startswith("win") else "gcc"
        params["CC"] = os.getenv("CC", default_cc)
    if "CXX" not in params:
        default_cx = "g++.exe" if sys.platform.startswith("win") else "g++"
        params["CXX"] = os.getenv("CXX", default_cx)
    if "OS_SIZE" not in params or "OS_ARCH" not in params:
        default_size = "64" if sys.maxsize > 2**32 else "32"
        default_arch = "x86_64" if default_size == "64" else "x86"
        params["OS_SIZE"] = os.getenv("OS_SIZE", default_size)
        params["OS_ARCH"] = os.getenv("OS_ARCH", default_arch)
    if "DEBUG" not in params:
        params["DEBUG"] = None
    if "CUDA" not in params:
        params["CUDA"] = False
    # TODO: More intelligently create dependency object string
    params["myriad_lib_objs"] =\
        " ".join([dep.__name__ + ".o" for dep in dependencies])
    return params


class MyriadSimul(_MyriadSimulParent, metaclass=_MyriadSimulMeta):
    """
    Myriad Simulation object, holding object state and communicating with
    simulation process, including saving data.
    """

    def __init__(self, compartments: set=None, mechanisms: set=None, **kwargs):
        #: Internal global compartment list
        self._compartments = compartments if compartments else set()
        #: Internal global mechanism list
        self._mechanisms = mechanisms if mechanisms else set()
        #: Simulation parameters as a dictionary, automatically-filled
        self.simul_params = _setup_simul_params(kwargs,
                                                getattr(self, "dependencies"))
        #: Template for Makefile, used for building C executable
        self._makefile_template = MakoFileTemplate("Makefile",
                                                   MAKEFILE_TEMPLATE,
                                                   self.simul_params)
        #: Template for setup.py, used for building CPython extensions
        self._setuppy_template = MakoFileTemplate(
            "setup.py",
            SETUPPY_TEMPLATE,
            {"dependencies": getattr(self, "dependencies")})

    def add_mechanism(self, comp, mech):
        """ 'Adds' the mechanism to the compartment """
        if comp is None:
            raise ValueError("Cannot add a mechanism to a null Compartment")
        elif mech is None:
            raise ValueError("Cannot add a null mechanism to a Compartment")
        # TODO: Do additional overhead work to 'link' mechanism to compartment
        self._mechanisms.add(mech)

    def add_compartment(self, comp):
        """ 'Adds' the compartment to the global compartment list """
        self._compartments.add(comp)

    def setup(self):
        """ Creates and links Compartments and mechanisms """
        raise NotImplementedError("Please override setup() in your class")

    def run(self):
        """ Runs the simulation and puts results back into Python objects """
        if len(self._compartments) == 0:
            raise RuntimeError("No compartments found!")
        self._makefile_template.render_to_file()
        self._setuppy_template.render_to_file()

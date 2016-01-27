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

#: Template for main.c (main executable C file)
MAIN_TEMPLATE = resource_string(__name__,
                                "templates/main.c.mako").decode("UTF-8")

#: Template for myriad.h (main parameter/macro file)
MYRIAD_H_TEMPLATE = resource_string(__name__,
                                    "templates/myriad.h.mako").decode("UTF-8")

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
    if "FAST_EXP" not in params:
        params["FAST_EXP"] = False
    if "MYRIAD_ALLOCATOR" not in params:
        params["MYRIAD_ALLOCATOR"] = False
    if "NUM_THREADS" not in params:
        params["NUM_THREADS"] = 1
    # TODO: More intelligently create dependency object string
    params["myriad_lib_objs"] =\
        " ".join([dep.__name__ + ".o" for dep in dependencies])
    params["dependencies"] = dependencies
    return params


class MyriadSimul(_MyriadSimulParent, metaclass=_MyriadSimulMeta):
    """
    Myriad Simulation object, holding object state and communicating with
    simulation process, including saving data.
    """

    def __init__(self,
                 compartments: set=None,
                 mechanisms: set=None,
                 dt: float=0.01,
                 simul_len: int=1000,
                 **kwargs):
        #: Internal global compartment list
        self._compartments = compartments if compartments else set()
        kwargs["NUM_CELLS"] = len(self._compartments)
        #: Internal global mechanism list
        self._mechanisms = mechanisms if mechanisms else set()
        # TODO: More intelligently calculate MAX_NUM_MECHS
        kwargs["MAX_NUM_MECHS"] = 32
        # TODO: Change DT to Units
        kwargs["DT"] = dt
        # TODO: Change SIMUL_LEN to Units
        kwargs["SIMUL_LEN"] = simul_len
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
        self._main_template = MakoFileTemplate("main.c",
                                               MAIN_TEMPLATE,
                                               self.simul_params)
        self._myriad_h_template = MakoFileTemplate("myriad.h",
                                                   MYRIAD_H_TEMPLATE,
                                                   self.simul_params)

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

    # TODO: Create temporary directory to put all this crap in
    def run(self):
        """ Runs the simulation and puts results back into Python objects """
        if len(self._compartments) == 0:
            raise RuntimeError("No compartments found!")
        # Render templates for dependencies
        for dependency in getattr(self, "dependencies"):
            dependency.render_templates()
        # Render templates for simulation-specific files
        self._makefile_template.render_to_file()
        self._setuppy_template.render_to_file()
        self._main_template.render_to_file()
        self._myriad_h_template.render_to_file()
        # Once templates are rendered, perform compilation
        subprocess.check_call(["make", "all"])

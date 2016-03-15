"""
Myriad Simulation Object
"""

import os
import sys
import logging
import subprocess
import importlib
import time

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

#: Template for pymyriad.h (main CPython interface file)
PYMYRIAD_H_TEMPLATE = resource_string(
    __name__,
    "templates/pymyriad.h.mako").decode("UTF-8")

#: Template for myriad_alloc.c (myriad memory allocator utility implementation)
MYRIAD_ALLOC_C_TEMPLATE = resource_string(
    __name__,
    "templates/myriad_alloc.c.mako").decode("UTF-8")

#: Template for myriad_alloc.h (myriad memory allocator utility header)
MYRIAD_ALLOC_H_TEMPLATE = resource_string(
    __name__,
    "templates/myriad_alloc.h.mako").decode("UTF-8")

#: Template for myriad_communicator.c (myriad UDP socket API for IPC impl)
MYRIAD_COMMUNICATOR_C_TEMPLATE = resource_string(
    __name__,
    "templates/myriad_communicator.c.mako").decode("UTF-8")

#: Template for myriad_communicator.c (myriad UDP socket API for IPC header)
MYRIAD_COMMUNICATOR_H_TEMPLATE = resource_string(
    __name__,
    "templates/myriad_communicator.h.mako").decode("UTF-8")

#: Template for pmyriad.c (myriad Python 'glue' for object interpretation)
PYMYRIAD_C_TEMPLATE = resource_string(
    __name__,
    "templates/pymyriad.c.mako").decode("UTF-8")

#: Template for pymyriad_commuinicator.c (myriad Python 'glue' for IPC)
PYMYRIAD_COMMUNICATOR_C_TEMPLATE = resource_string(
    __name__,
    "templates/pymyriad_communicator.c.mako").decode("UTF-8")

#######
# Log #
#######

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

###########
# Classes #
###########


class SubprocessCommunicator(object):
    """
    Communicator that manages the connection with the Myriad subprocess
    """

    def __init__(self, myriad_comm_mod, binary_rel_path: str="/main.bin"):
        #: Child process
        self.child_proc = None
        #: Connection initialization status
        self.connected = False
        #: Myriad communicator dynamically-loaded module
        self.myriad_comm_mod = myriad_comm_mod
        if self.myriad_comm_mod is None:
            raise ValueError("Myriad communicator module may not be None")
        #: Myriad binary relative path
        self.binary_rel_path = binary_rel_path

    def spawn_child(self):
        """ Spawns subprocess executable """
        self.child_proc = subprocess.Popen(
            [os.getcwd() + self.binary_rel_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)

    def setup_connection(self):
        """ Waits for the subprocess to connect to the socket """
        if self.child_proc is None:
            raise RuntimeError("Child process is not yet running")
        self.myriad_comm_mod.init()
        self.connected = True

    def request_data(self, obj_id: int) -> object:
        """ Requests data from subprocess and returns a Compartment object """
        if self.child_proc is None:
            raise RuntimeError("Child process is not yet running")
        elif self.connected is False:
            raise RuntimeError("Not connected to child process")
        elif obj_id < 0:
            raise ValueError("Invalid obj_id value: value out of range")
        return self.myriad_comm_mod.retrieve_obj(obj_id)

    def close_connection(self):
        """ Closes the connection to the subprocess """
        # Ask child process to terminate
        self.child_proc.poll()
        if self.child_proc.returncode is None:
            self.child_proc.terminate()
        self.child_proc = None
        self.connected = False


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
    if "RANDOM_SEED" not in params:
        params["RANDOM_SEED"] = 42  # FIXME: Use time() for default seed
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
        self._compartments = compartments if compartments else list()
        #: Internal global mechanism list of lists
        self._mechanisms = mechanisms if mechanisms else list([])
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
        self._makefile_tmpl = None
        #: Template for setup.py, used for building CPython extensions
        self._setuppy_tmpl = None
        #: Template for main.c main simulation file
        self._main_tmpl = None
        #: Template for myriad.h parameter and macro file
        self._myriad_h_tmpl = None
        #: Template for pymyriad.h CPython interface header
        self._pymyriad_h_tmpl = None
        #: Template for myriad_alloc.c myriad memory allocator utility
        self._myriad_alloc_c_tmpl = None
        #: Template for myriad_alloc.h myriad memory allocator utility header
        self._myriad_alloc_h_tmpl = None
        #: Template for myriad_communicator.c myriad UDP socket API for IPC
        self._myriad_communicator_c_tmpl = None
        #: Template for myriad_communicator.h myriad UDP socket API for IPC
        self._myriad_communicator_h_tmpl = None
        #: Template for pmyriad.c myriad Python 'glue' for object interp
        self._pymyriad_c_tmpl = None
        #: Template for pymyriad_commuinicator.c myriad Python 'glue' for IPC
        self._pymyriad_communicator_c_tmpl = None

    def add_mechanism(self, comp, mech):
        """ 'Adds' the mechanism to the compartment """
        if comp is None:
            raise ValueError("Cannot add a mechanism to a null Compartment")
        elif mech is None:
            raise ValueError("Cannot add a null mechanism to a Compartment")
        elif comp not in self._compartments:
            self._compartments.append(comp)
            self._mechanisms.append([])
        # Mechanism is added based on what compartment its in
        indx = self._compartments.index(comp)
        if mech in self._mechanisms[indx]:
            raise ValueError("Mechanism was already added to a Compartment")
        self._mechanisms[indx].append(mech)

    def add_compartment(self, comp):
        """ 'Adds' the compartment to the global compartment list """
        if comp is None:
            raise ValueError("Cannot add a null Compartment")
        elif comp in self._compartments:
            raise ValueError("Compartment has already been added")
        self._compartments.append(comp)

    def setup(self):
        """ Creates and links Compartments and mechanisms """
        raise NotImplementedError("Please override setup() in your class")

    # TODO: Create temporary directory to put all this crap in
    def run(self):
        """ Runs the simulation and puts results back into Python objects """
        # Calculate the number of compartments
        if len(self._compartments) == 0:
            raise RuntimeError("No compartments found!")
        self.simul_params["NUM_COMPARTMENTS"] = len(self._compartments)
        # Render templates for dependencies
        for dependency in getattr(self, "dependencies"):
            dependency.render_templates()
        # Create & render templates for simulation-specific files
        self._makefile_tmpl = MakoFileTemplate(
            "Makefile",
            MAKEFILE_TEMPLATE,
            self.simul_params)
        self._setuppy_tmpl = MakoFileTemplate(
            "setup.py",
            SETUPPY_TEMPLATE,
            {"dependencies": getattr(self, "dependencies")})
        main_params = {"compartments": self._compartments,
                       "mechanisms": self._mechanisms}
        main_params.update(self.simul_params)
        self._main_tmpl = MakoFileTemplate(
            "main.c",
            MAIN_TEMPLATE,
            main_params)
        self._myriad_h_tmpl = MakoFileTemplate(
            "myriad.h",
            MYRIAD_H_TEMPLATE,
            self.simul_params)
        self._pymyriad_h_tmpl = MakoFileTemplate(
            "pymyriad.h",
            PYMYRIAD_H_TEMPLATE,
            self.simul_params)
        self._myriad_alloc_c_tmpl = MakoFileTemplate(
            "myriad_alloc.c",
            MYRIAD_ALLOC_C_TEMPLATE,
            self.simul_params)
        self._myriad_alloc_h_tmpl = MakoFileTemplate(
            "myriad_alloc.h",
            MYRIAD_ALLOC_H_TEMPLATE,
            self.simul_params)
        self._myriad_communicator_c_tmpl = MakoFileTemplate(
            "myriad_communicator.c",
            MYRIAD_COMMUNICATOR_C_TEMPLATE,
            self.simul_params)
        self._myriad_communicator_h_tmpl = MakoFileTemplate(
            "myriad_communicator.h",
            MYRIAD_COMMUNICATOR_H_TEMPLATE,
            self.simul_params)
        self._pymyriad_c_tmpl = MakoFileTemplate(
            "pymyriad.c",
            PYMYRIAD_C_TEMPLATE,
            self.simul_params)
        self._pymyriad_communicator_c_tmpl = MakoFileTemplate(
            "pymyriad_communicator.c",
            PYMYRIAD_COMMUNICATOR_C_TEMPLATE,
            self.simul_params)
        # Render templates to file
        self._makefile_tmpl.render_to_file()
        self._setuppy_tmpl.render_to_file()
        self._main_tmpl.render_to_file()
        self._myriad_h_tmpl.render_to_file()
        self._pymyriad_h_tmpl.render_to_file()
        self._myriad_alloc_c_tmpl.render_to_file()
        self._myriad_alloc_h_tmpl.render_to_file()
        self._myriad_communicator_c_tmpl.render_to_file()
        self._myriad_communicator_h_tmpl.render_to_file()
        self._pymyriad_c_tmpl.render_to_file()
        self._pymyriad_communicator_c_tmpl.render_to_file()
        # Once templates are rendered, perform compilation
        subprocess.check_call(["make", "-j4", "all"])
        # Invalidate cache and load dynamic extensions
        # TODO: Change this path to something platform-specific (autodetect)
        sys.path.append("build/lib.linux-x86_64-3.4/")
        importlib.invalidate_caches()
        myriad_comm_mod = importlib.import_module("myriad_comm")
        for dependency in getattr(self, "dependencies"):
            if dependency.__name__ != "MyriadObject":
                importlib.import_module(dependency.__name__.lower())
        # Run simulation and return the communicator object back
        comm = SubprocessCommunicator(myriad_comm_mod)
        comm.spawn_child()
        time.sleep(0.25)  # FIXME: Change this sleep to a wait of some kind
        comm.setup_connection()
        return comm

#!/usr/bin/env python
"""
Compile
"""
from distutils.core import setup, Extension  # pylint: disable=E0611,F0401
from distutils.version import StrictVersion  # pylint: disable=E0611,F0401
from distutils.unixccompiler import UnixCCompiler  # pylint: disable=E0611,F0401
from numpy import get_include as np_includes
from numpy.version import version as np_version

# We need Numpy 1.8 or greater
assert StrictVersion(np_version) > StrictVersion("1.8")

# Create the static library
COMPILER = UnixCCompiler(verbose=2)

MYRIAD_CPYTHON_DEFS = [("_POSIX_C_SOURCE", "200809L"),
                       ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

MYRIAD_CPYTHON = Extension("mmqpy",
                           define_macros=MYRIAD_CPYTHON_DEFS,
                           extra_compile_args=["-std=gnu99"],
                           include_dirs=["/usr/include", np_includes()],
                           library_dirs=["/usr/lib/"],
                           libraries=["rt", "pthread"],
                           sources=['mmqpy.c', 'mmq.c'])

NODDY = Extension("noddy",
                  define_macros=MYRIAD_CPYTHON_DEFS,
                  extra_compile_args=["-std=gnu99"],
                  include_dirs=["/usr/include", np_includes()],
                  library_dirs=["/usr/lib/"],
                  libraries=["rt", "pthread"],
                  sources=["noddy.c"])

SPAMDICT = Extension("spamdict",
                     define_macros=MYRIAD_CPYTHON_DEFS,
                     extra_compile_args=["-std=gnu99"],
                     include_dirs=["/usr/include", np_includes()],
                     library_dirs=["/usr/lib/"],
                     libraries=["rt", "pthread"],
                     sources=["spamdict.c"])

PYMYRIADOBJECT = Extension("pymyriadobject",
                           define_macros=MYRIAD_CPYTHON_DEFS,
                           extra_compile_args=["-std=gnu99"],
                           include_dirs=["/usr/include", np_includes()],
                           library_dirs=["/usr/lib/"],
                           libraries=["rt", "pthread"],
                           sources=["pymyriadobject.c"])

PYCOMPARTMENT = Extension("pycompartment",
                          define_macros=MYRIAD_CPYTHON_DEFS,
                          extra_compile_args=["-std=gnu99"],
                          include_dirs=["/usr/include", np_includes()],
                          library_dirs=["/usr/lib/"],
                          libraries=["rt", "pthread"],
                          sources=["pycompartment.c"])

setup(name="mmqpy",
      version="1.0",
      description="Python message queue package",
      ext_modules=[MYRIAD_CPYTHON,
                   NODDY,
                   SPAMDICT,
                   PYMYRIADOBJECT,
                   PYCOMPARTMENT])

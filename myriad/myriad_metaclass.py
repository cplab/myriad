""".. module:: myriad_metaclass
:platform: Linux
:synopsis: Provides metaclass for automatic Myriad integration
.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>

Purpose of this module is to provide an abstraction layer for Myriad objects,
classes, and the methods belonging to them. This is done so that inheritance
and method overloading can be done as a pre-compilation layer, instead of
using C++ classes which cannot be unconditionally used in CUDA code.

TODO: Add details about the CUDAfication process for classes/objects/methods.

The following abstractions are expanded upon here:

=======
Objects
=======

Myriad's objects are structured as simple structs with state variables. We
expect the compiler to reasonably align the struct so that binary compatibility
is maintained across platforms. Struct declarations are done in a header file
named after the object, e.g. 'MyriadObject.cuh'.

Inheriting another Myriad object's state is a simple manner of embedding it::

    struct ChildObject {
        struct ParentObject _;  // Embedded parent object
        int my_state;
    };

This is done because you can then up-cast objects for free, since the memory
start of the parent struct is the same as the memory start of the child::

    struct ParentObject* parent = (struct ParentObject*) child_struct_ptr;
    parent->parent_state++;  // This will alter *child_struct_ptr

Note that this does not work in the reverse, i.e. down-casts.

All objects are 'inherited' from `struct MyriadObject`, which itself contains
an enum identifying the object's class (see :mod:`MyriadObject` for details)

=======
Classes
=======

Classes are not actual objects like they are in other object-oriented
programming languages. Instead, an object can have any number virtual methods,
which are inheritable/overridable by child objects. These are simply written as
globally-visible functions, declared in the object's header file.

Each object type is created with an enum value identifying its class embedded
in the state of the originator object (see :mod:`MyriadObject` for details on
how this is accomplished via `myriad_new`).

Thus, the entire parent object can be represented as a single enum struct:

    // Class enum values
    enum MyriadClass
    {
        MYRIADOBJECT = 0,
        MECHANISM,
        COMPARTMENT,
        ...
        NUM_CU_CLASS
    };

    // Progenitor object
    typedef struct MyriadObject
    {
        const enum MyriadClass class_id;
    } *MyriadObject_t;

The class enum value is chiefly used for virtual table lookups in the functions
that are used to generically call virtual functions, as follows:

    // Function for calling an object's constructor
    inline void myriad_ctor(obj, vap) {
        ctor_vtable[((MyriadObject_t) obj)->class_id](obj, vap);
    }

Classes' virtual functions are initialized at runtime via a special stand-alone
`init` function that dynamically allocates the virtual tables. This allows for
reduced overhead when only a small subset of classes are required in the
simulation kernel.

See the section below for how methods are abstracted.

=======
Methods
=======

Methods are considered to be a loose amalgamation of exactly three function
definitions and a function pointer typedef (for virtual tables):

1. Instance method, i.e. the implementor of the method
2. Delegator function, which appropriately derefences the appropriate vtable
3. Super Delegator function, which does the same but for the parent class

Each delegator template can be see in :class:`MyriadMethod`, but in short the
purpose of those functions are to acquire an object's class enum and
dereference the appropriate function pointer from the virtual function table.

The same applies for the 'super' delegator, albeit it will call the
given object's superclass' version of the function instead. This is provided
so that constructors and destructor calls may be made recursively up the
inheritance tree.

"""

import inspect
import logging
import os

from collections import OrderedDict
from copy import copy
from functools import wraps
from pkg_resources import resource_string

from .myriad_mako_wrapper import MakoTemplate
from .myriad_utils import OrderedSet
from .myriad_types import MyriadScalar, MyriadFunction, MyriadStructType
from .myriad_types import _MyriadBase, MyriadCType, MyriadTimeseriesVector
from .myriad_types import MVoid, MInt, MDouble, filter_inconvertible_types
from .ast_function_assembler import pyfun_to_cfun

#######
# Log #
#######

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

#############
# Constants #
#############

# Default include headers for all C files
DEFAULT_LIB_INCLUDES = {"stdlib.h",
                        "stdio.h",
                        "assert.h",
                        "string.h",
                        "stddef.h",
                        "stdarg.h",
                        "stdint.h"}

# Default include headers for CUDA files
DEFAULT_CUDA_INCLUDES = {"cuda_runtime.h", "cuda_runtime_api.h"}


#############
# Templates #
#############

DELG_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "delegator_func.mako")).decode("UTF-8")

SUPER_DELG_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "super_delegator_func.mako")).decode("UTF-8")


######################
# Delegator Creation #
######################


def create_delegator(instance_fxn: MyriadFunction,
                     classname: str) -> MyriadFunction:
    """
    Creates a delegator function based on a function definition.

    :param MyriadFunction instance_fxn: Instance function to be wrapped
    :param str classname: Name of the class delegator
    :return: New MyriadFunction representing the delegator around instance_fxn
    :rtype: MyriadFunction
    """
    # Create copy with modified identifier
    ist_cpy = MyriadFunction.from_myriad_func(instance_fxn)
    # Generate template and render into copy's definition
    template_vars = {"delegator": ist_cpy,
                     "classname": classname,
                     "MVoid": MVoid}
    template = MakoTemplate(DELG_TEMPLATE, template_vars)
    LOG.debug("Rendering create_delegator template for %s", classname)
    template.render()
    ist_cpy.fun_def = template.buffer
    # Return created copy
    return ist_cpy


def create_super_delegator(delg_fxn: MyriadFunction,
                           classname: str) -> MyriadFunction:
    """
    Create super delegator function.

    :param MyriadFunction delg_fxn: Delegator to create super_* wrapper for
    :param str classname: Name of the base class for this super delegator

    :return: Super delegator method as a MyriadFunction
    :rtype: MyriadFunction
    """
    # Create copy of delegator function with modified parameters
    super_args = copy(delg_fxn.args_list)
    super_class_arg = MyriadScalar("_class", MInt, False, ["const"])
    tmp_arg_indx = len(super_args) + 1
    super_args[tmp_arg_indx] = super_class_arg
    super_args.move_to_end(tmp_arg_indx, last=False)
    s_delg_f = MyriadFunction.from_myriad_func(delg_fxn,
                                               "super_" + delg_fxn.ident,
                                               super_args)
    # Generate template and render
    template_vars = {"delegator": delg_fxn,
                     "super_delegator": s_delg_f,
                     "classname": classname,
                     "MVoid": MVoid}
    template = MakoTemplate(SUPER_DELG_TEMPLATE, template_vars)
    LOG.debug("Rendering create_super_delegator template for %s", classname)
    template.render()
    # Add rendered definition to function
    s_delg_f.fun_def = template.buffer
    return s_delg_f


def gen_instance_method_from_str(delegator, m_name: str,
                                 method_body: str) -> MyriadFunction:
    """
    Automatically generate a MyriadFunction wrapper for a method body.

    :param str m_name: Name to prepend to the instance method identifier
    :param str method_body: String template to use as the method body

    :return: Instance method as a MyriadFunction
    :rtype: MyriadFunction
    """
    return MyriadFunction(m_name + '_' + delegator.ident,
                          args_list=delegator.args_list,
                          ret_var=delegator.ret_var,
                          fun_def=method_body)

#####################
# Method Decorators #
#####################


def myriad_method(method):
    """
    Tags a method in a class to be a myriad method (i.e. converted to a C func)
    NOTE: This MUST be the first decorator applied to the function! E.g.:
    `
    @another_decorator
    @yet_another_decorator
    @myriad_method
    def my_fn(stuff):
    `
    This is because decorators replace the wrapped function's signature.
    """
    @wraps(method)
    def inner(*args, **kwargs):
        """ Dummy inner function to prevent direct method calls """
        raise Exception("Cannot directly call a myriad method")
    LOG.debug("myriad_method annotation wrapping %s", method.__name__)
    setattr(inner, "is_myriad_method", True)
    setattr(inner, "original_fun", method)
    return inner


def myriad_method_verbatim(method):
    """
    Tags a method in a class to be a myriad method (i.e. converted to a C func)
    but takes the docstring as verbatim C code.

    NOTE: This MUST be the first decorator applied to the function! E.g.:
    `
    @another_decorator
    @yet_another_decorator
    @myriad_method_verbatim
    def my_fn(stuff):
    `

    This is because decorators replace the wrapped function's signature.
    """
    @wraps(method)
    def inner(*args, **kwargs):
        """ Dummy inner function to prevent direct method calls """
        raise Exception("Cannot directly call a myriad method")
    setattr(inner, "is_myriad_method_verbatim", True)
    setattr(inner, "is_myriad_method", True)
    setattr(inner, "original_fun", method)
    return inner


def _myriadclass_method(method):
    """
    Tags a method in a class to be a MyriadClass method.

    MyriadClass methods are methods exclusive to MyriadClass; they are not
    declared as part of the MyriadClass struct but are used internally in
    MyriadObject.c to define behaviour tied to MyriadObject inheritance.

    NOTE: This MUST be the first decorator applied to the function! E.g.:
    `
    @another_decorator
    @yet_another_decorator
    @_myriadclass_method
    def my_fn(stuff):
    `

    This is because decorators replace the wrapped function's signature.
    """
    @wraps(method)
    def inner(*args, **kwargs):
        """ Dummy inner function to prevent direct method calls """
        raise Exception("Cannot directly call a myriad method")
    LOG.debug("myriad_method annotation wrapping %s", method.__name__)
    setattr(inner, "is_myriad_method_verbatim", True)
    setattr(inner, "is_myriad_method", True)
    setattr(inner, "is_myriadclass_method", True)
    setattr(inner, "original_fun", method)
    return inner

#####################
# MetaClass Wrapper #
#####################


class _MyriadObjectBase(object):
    """ Dummy placeholder class used for type checking, circular dependency"""

    @classmethod
    def _fill_in_base_methods(cls,
                              child_namespace: OrderedDict,
                              myriad_methods: OrderedDict):
        """
        Fills in missing base methods (e.g. ctor/etc) in child's namespace
        """
        # raise NotImplementedError("Not implemented in _MyriadObjectBase")
        pass

    @classmethod
    def get_file_list(cls) -> list:
        """ Generates list of files generated by Myriad for this class """
        base_name = cls.__name__
        return [base_name + ".cuh",
                base_name + ".cu",
                "py" + base_name.lower() + ".c"]


def _method_organizer_helper(supercls: _MyriadObjectBase,
                             myriad_methods: OrderedDict) -> OrderedSet:
    """
    Organizes Myriad Methods, including inheritance and verbatim methods.

    Verbatim methods are converted differently than pythonic methods; their
    docstring is embedded 'verbatim' into the template instead of going through
    the full AST conversion (though the function header is still processed).

    Returns an OrderedSet of methods not defined in the superclass
    """
    # Convert methods; remember, items() returns a read-only view
    for m_ident, method in myriad_methods.items():
        # Process verbatim methods
        verbatim = hasattr(method, "is_myriad_method_verbatim")
        # Check if verbatim methods have a docstring to use
        if verbatim and (method.__doc__ is None or method.__doc__ == ""):
            raise Exception("Verbatim method cannot have empty docstring")
        # Parse method, converting the body only if not verbatim
        myriad_methods[m_ident] = pyfun_to_cfun(method.original_fun, verbatim)
        # TODO: Use local var to avoid adding to own_methods (instead of attr)
        if hasattr(method, "is_myriadclass_method"):
            setattr(myriad_methods[m_ident], "is_myriadclass_method", True)

    # The important thing here is to decide which methods
    # (1) WE'VE CREATED, and
    # (2) Which methods are being OVERRRIDEN BY US that ORIGINATED ELSEWHERE
    def get_parent_methods(cls: _MyriadObjectBase) -> OrderedSet:
        """ Gets the own_methods of the parent class, and its parents, etc. """
        # If we're MyriadObject, we don't have any parent methods
        if cls is _MyriadObjectBase:
            return OrderedSet()
        else:
            return cls.own_methods.union(get_parent_methods(cls.__bases__[0]))

    # Get parent method IDENTIFIERS - easier to check for existence
    parent_methods = set([v.ident for v in get_parent_methods(supercls)])
    LOG.debug("_method_organizer_helper parent methods: %r", parent_methods)

    # 'Own methods' are methods we've created (1); everything else is (2)
    own_methods = OrderedSet()
    for m_ident, mtd in myriad_methods.items():
        if m_ident in parent_methods or hasattr(mtd, "is_myriadclass_method"):
            continue
        own_methods.add(mtd)

    LOG.debug("_method_organizer_helper own methods selected: %r", own_methods)
    return own_methods


def _generate_includes(superclass) -> (set, set):
    """ Generates local and lib includes based on superclass """
    lcl_inc = []
    if superclass is not _MyriadObjectBase:
        lcl_inc = [superclass.__name__ + ".h"]
    # TODO: Better detection of system/library headers
    lib_inc = copy(DEFAULT_LIB_INCLUDES)
    return (lcl_inc, lib_inc)


def _generate_cuda_includes(superclass) -> (set, set):
    """ Generates local and lib includes for a CUDA file """
    lcl_inc = []
    if superclass is not _MyriadObjectBase:
        lcl_inc = [superclass.__name__ + ".cuh"]
    return (lcl_inc, DEFAULT_LIB_INCLUDES | DEFAULT_CUDA_INCLUDES)


def _parse_namespace(namespace: dict,
                     name: str,
                     myriad_methods: OrderedDict,
                     myriad_obj_vars: OrderedDict):
    """
    Parses the given namespace, updates the last three input arguments to have:
        1) OrderedDict of myriad_methods
        2) OrderedDict of myriad_obj_vars
        3) OrderedSet of verbatim methods
    """
    # Extracts variables and myriad methods from class definition
    for k, val in namespace.items():
        # if val is ...
        # ... a registered myriad method
        if hasattr(val, "is_myriad_method"):
            if hasattr(val, "is_myriadclass_method"):
                LOG.debug("%s is a MyriadClass method in %s", k, name)
            elif hasattr(val, "is_myriad_method_verbatim"):
                LOG.debug("%s is a verbatim myriad method in %s", k, name)
            else:
                LOG.debug("%s is a myriad method in %s", k, name)
            myriad_methods[k] = val
        # ... some generic non-Myriad function or method
        elif inspect.isfunction(val) or inspect.ismethod(val):
            LOG.debug("%s is a function or method, ignoring for %s", k, name)
        # ... some generic instance of a _MyriadBase type
        elif issubclass(val.__class__, _MyriadBase):
            LOG.debug("%s is a Myriad-type non-function attribute", k)
            myriad_obj_vars[k] = val
            LOG.debug("%s was added as an object variable to %s", k, name)
        # ... a type statement of base type MyriadCType (e.g. MDouble)
        elif issubclass(val.__class__, MyriadCType):
            myriad_obj_vars[k] = MyriadScalar(k, val)
            LOG.debug("%s has decl %s", k, myriad_obj_vars[k].stringify_decl())
            LOG.debug("%s was added as an object variable to %s", k, name)
        # ... a timeseries variable
        elif val is MyriadTimeseriesVector:
            # TODO: Enable different precisions for MyriadTimeseries
            myriad_obj_vars[k] = MyriadScalar(k, MDouble, arr_id="SIMUL_LEN")
        # ... a python meta value (e.g.  __module__) we shouldn't mess with
        elif k.startswith("__"):
            LOG.debug("Built-in method %r ignored for %s", k, name)
        # TODO: Figure out other valid values for namespace variables
        else:
            LOG.info("Unsupported var type for %r, ignoring in %s", k, name)
    LOG.debug("myriad_obj_vars for %s: %s ", name, myriad_obj_vars)


class MyriadMetaclass(type):
    """
    TODO: Documentation for MyriadMetaclass
    """
    #: Master list of all MyriadClasses
    myriad_classes = OrderedSet()

    @classmethod
    def __prepare__(mcs, name, bases):
        """
        Force the class to use an OrderedDict as its __dict__, for purposes of
        enforcing strict ordering of keys (necessary for making structs).
        """
        return OrderedDict()

    @staticmethod
    def myriad_init(self, **kwargs):
        """ Initializes the Myriad Object """
        def _get_obj_vars(obj):
            """ Accrues all object variables up the object inheritance tree """
            if not hasattr(obj, "myriad_obj_vars"):
                return {}
            obj_vars = OrderedDict()
            if obj.__class__.__bases__:
                obj_vars.update(_get_obj_vars(obj.__class__.__bases__[0]))
            obj_vars.update(getattr(obj, "myriad_obj_vars"))
            return obj_vars
        # Filter out obj_vars for non-desirable struct types
        expected_obj_vars = filter_inconvertible_types(_get_obj_vars(self))
        LOG.debug("obj_vars: %s", expected_obj_vars)
        # If values are missing, error out
        set_diff = set(expected_obj_vars.keys()) ^ set(kwargs.keys())
        if set_diff:
            slen = len(expected_obj_vars.keys()) - len(kwargs.keys())
            msg = "Too {0} arguments for {1} __init__: {2} {3}"
            raise ValueError(msg.format("few" if slen > 0 else "many",
                                        self.__class__,
                                        "missing" if slen > 0 else "extra",
                                        set_diff))
        # Assign all valid values
        for argname, argval in kwargs.items():
            setattr(self, argname, argval)
        # Store object vars in object to use for myriad_new
        setattr(self, "myriad_new_params", expected_obj_vars)

    @staticmethod
    def myriad_set_attr(self, argname, argval):
        """
        Prevent users from accessing objects except through py_x interfaces
        """
        LOG.warning("Myriad Object attributes not fully suppported!")
        self.__dict__[argname] = argval

    def __new__(mcs, name, bases, namespace, **kwds):
        if len(bases) > 1:
            raise NotImplementedError("Multiple inheritance is not supported.")
        if len(kwds) > 1:
            raise NotImplementedError("Extra paremeters not supported.")

        supercls = bases[0]  # Alias for base class
        if not issubclass(supercls, _MyriadObjectBase):
            raise TypeError("Myriad modules must inherit from MyriadObject")

        # Setup object/class variables, methods, and verbatim methods
        myriad_obj_vars = OrderedDict()
        myriad_methods = OrderedDict()

        # Setup object with implicit superclass to start of struct definition
        if supercls is not _MyriadObjectBase:
            myriad_obj_vars["_"] = supercls.obj_struct("_", quals=["const"])

        # Parse namespace into appropriate variables
        _parse_namespace(namespace,
                         name,
                         myriad_methods,
                         myriad_obj_vars)

        # Object Name and Class Name are automatically derived from name
        namespace["obj_name"] = name
        namespace["cls_name"] = name + "Class"

        # Struct definition representing object state
        namespace["obj_struct"] = MyriadStructType(namespace["obj_name"],
                                                   myriad_obj_vars)

        # Organize myriad methods and class struct members
        namespace["own_methods"] = _method_organizer_helper(supercls,
                                                            myriad_methods)

        # Add #include's from system libraries, local files, and CUDA headers
        namespace["local_includes"], namespace["lib_includes"] =\
            _generate_includes(supercls)
        namespace["cuda_local_includes"], namespace["cuda_lib_includes"] =\
            _generate_cuda_includes(supercls)

        # Add other objects to namespace
        namespace["myriad_methods"] = myriad_methods
        namespace["myriad_obj_vars"] = myriad_obj_vars
        namespace["init_functions"] = ""

        # Fill in missing methods (ctor/etc.)
        supercls._fill_in_base_methods(namespace, myriad_methods)

        # Finally, delete function from namespace
        for method_id in myriad_methods.keys():
            if method_id in namespace:
                del namespace[method_id]

        # Generate internal module representation
        namespace["__init__"] = MyriadMetaclass.myriad_init
        namespace["__setattr__"] = MyriadMetaclass.myriad_set_attr
        new_type = type.__new__(mcs, name, (supercls,), dict(namespace))

        # Register implementor class
        MyriadMetaclass.myriad_classes.add(new_type)

        return new_type

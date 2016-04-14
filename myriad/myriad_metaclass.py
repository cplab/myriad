"""
.. module:: myriad_metaclass
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
named after the object, e.g. 'MyriadObject.h'.

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
a pointer to the object's class singleton (see :mod:`MyriadObject` for details)

=======
Classes
=======

Classes are structured identically to objects; in fact, classes are themselves
objects (see :mod:`MyriadObject` for how this circular dependency is resolved).

The major difference is that while objects' structs are replicable state, so
that they mimic other OOP models with instances having their own, separable
state, classes are meant to be treated as define-once-reference-everywhere
singletons. Each object type is created with a pointer to its class definition
embedded in the state of the originator object (see :mod:`MyriadObject` for
details on how this is accomplished via `myriad_new`).

Classes' own state are composed entirely of function pointers representing
method definitions and the superclass they inherit from, for example::

    // Constructor function pointer type
    typedef void* (* ctor_t) (void* self, va_list* app);

    struct ChildClass {
        struct ParentClass _;  // Embedded parent class
        ctor_t my_ctor;        // 'Method' storage using function pointer type
    }

Classes are initialized on-demand at runtime via a special stand-alone `init`
function that dynamically creates the singletons. This allows for reduced
overhead when only a small subset of classes are required in the simulation
kernel, as well as allowing for run-time overriding of methods in a standard
fashion (i.e. by passing different arguments to the `myriad_new` calls that
create the class singletons).

See the section below for how methods are abstracted.

=======
Methods
=======

Methods are considered to be a loose amalgamation of 3 or more function
definitions in addition to a function pointer typedef (for class storage):

1. Stand-alone (i.e. not stored in class struct) Delegator function.
2. Stand-alone (i.e. not stored in class struct) Superclass Delegator function.
3. Any number of class-specific Instance Method function definitions.

Each delegator function template can be see in :class:`MyriadMethod`, but in
short the purpose of those functions are to acquire an object's class pointer
and dereference the class' function pointer. This is necessary because, since
the delegators are stand-alone functions with external bindings provided in
the object's header file, any code that includes said header file will be able
to call the delegator on any eligible (i.e. subclass'ed) object.

The same applies for the Super Delegator function, albeit it will call the
given object's superclass' version of the function instead. This is provided
so that constructors and destructor calls may be made recursively up the
inheritance tree.

Classes that declare new methods must provide an instance method definition
so that the `init` function (see above section about Classes) the constructor
can correctly override the right method. Subclasses can override methods in
the same fashion, provided they declare their own instance methods.
"""

import inspect
import logging
import os

from collections import OrderedDict
from copy import copy
from functools import wraps
from pkg_resources import resource_string

from pycparser.c_ast import ID, TypeDecl, Struct, PtrDecl, Decl, ArrayDecl

from .myriad_mako_wrapper import MakoTemplate, MakoFileTemplate

from .myriad_utils import OrderedSet

from .myriad_types import MyriadScalar, MyriadFunction, MyriadStructType
from .myriad_types import _MyriadBase, MyriadCType, MyriadTimeseriesVector
from .myriad_types import MDouble, MVoid, MSizeT, filter_inconvertible_types
from .myriad_types import c_decl_to_pybuildarg

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

INIT_OB_FUN_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "init_ob_fun.mako")).decode("UTF-8")

MYRIADOBJECT_OBJ_ARR_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "MyriadObject_obj_arr.mako")).decode("UTF-8")

MYRIADOBJECT_INIT_FUN_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "MyriadObject_obj_init_fun.mako")
).decode("UTF-8")

HEADER_FILE_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "header_file.mako")).decode("UTF-8")

CUH_FILE_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "cuda_header_file.mako")).decode("UTF-8")

CU_FILE_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "cuda_impl.mako")).decode("UTF-8")

C_FILE_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "c_file.mako")).decode("UTF-8")

PYC_COMP_FILE_TEMPLATE = resource_string(
    __name__,
    os.path.join("templates", "pyc_file.mako")).decode("UTF-8")

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
    super_class_arg = MyriadScalar("_class", MVoid, True, ["const"])
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
                          storage=['static'],
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
        return [base_name + ".c",
                base_name + ".h",
                base_name + ".cuh",
                base_name + ".cu",
                "py" + base_name.lower() + ".c"]


def _method_organizer_helper(supercls: _MyriadObjectBase,
                             myriad_methods: OrderedDict,
                             myriad_cls_vars: OrderedDict) -> OrderedSet:
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
        # For methods we've created, generate class variables for class struct
        own_methods.add(mtd)
        new_ident = "my_" + mtd.fun_typedef.name
        m_scal = MyriadScalar(new_ident, mtd.base_type)
        myriad_cls_vars[new_ident] = m_scal

    LOG.debug("_method_organizer_helper class variables selected: %r",
              myriad_cls_vars)
    return own_methods


def _template_creator_helper(namespace: OrderedDict) -> OrderedDict:
    """
    Creates templates using namespace, and returns the updated namespace.
    """
    # Initialize delegators/superdelegators in local namespace
    local_namespace = copy(namespace)
    own_method_delgs = []
    for method in namespace["own_methods"]:
        own_method_delgs.append(
            (create_delegator(method, namespace["cls_name"]),
             create_super_delegator(method, namespace["cls_name"])))
    local_namespace["own_method_delgs"] = own_method_delgs
    namespace["c_file_template"] = MakoFileTemplate(
        namespace["obj_name"] + ".c",
        C_FILE_TEMPLATE,
        local_namespace)
    LOG.debug("c_file_template done for %s", namespace["obj_name"])
    namespace["header_file_template"] = MakoFileTemplate(
        namespace["obj_name"] + ".h",
        HEADER_FILE_TEMPLATE,
        local_namespace)
    LOG.debug("header_file_template done for %s", namespace["obj_name"])
    namespace["cuh_file_template"] = MakoFileTemplate(
        namespace["obj_name"] + ".cuh",
        CUH_FILE_TEMPLATE,
        local_namespace)
    LOG.debug("cuh_file_template done for %s", namespace["obj_name"])
    namespace["cu_file_template"] = MakoFileTemplate(
        namespace["obj_name"] + ".cu",
        CU_FILE_TEMPLATE,
        local_namespace)
    LOG.debug("cu_file_template done for %s", namespace["obj_name"])
    # Initialize object struct conversion for CPython getter methods
    # Ignores superclass (_), class object, and array declarations
    # Places result in local namespace to avoid collisions/for efficiency
    pyc_scalar_types = {}
    for obj_var_name, obj_var_decl in namespace["obj_struct"].members.items():
        if (not obj_var_name.startswith("_") and
                not obj_var_name == "mclass" and
                not isinstance(obj_var_decl.type, ArrayDecl)):
            pyc_scalar_types[obj_var_name] = c_decl_to_pybuildarg(obj_var_decl)
    local_namespace["pyc_scalar_types"] = pyc_scalar_types
    namespace["pyc_file_template"] = MakoFileTemplate(
        "py" + namespace["obj_name"].lower() + ".c",
        PYC_COMP_FILE_TEMPLATE,
        local_namespace)
    LOG.debug("pyc_file_template done for %s", namespace["obj_name"])
    return namespace


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


def _init_module_vars(obj_name: str,
                      cls_name: str,
                      namespace: dict) -> OrderedDict:
    """ Special method for initializing MyriadObject objects"""
    module_vars = OrderedDict()
    is_myriad_obj = obj_name == "MyriadObject"
    # MyriadObject has that weird object array
    if is_myriad_obj:
        template = MakoTemplate(MYRIADOBJECT_OBJ_ARR_TEMPLATE, namespace)
        template.render()
        module_vars['object'] = template.buffer
    # Initialize const void pointers exported in module header files
    module_vars[obj_name] =\
        MyriadScalar(
            obj_name,
            MVoid,
            True,
            quals=["const"],
            init=ID("object") if is_myriad_obj else None)
    module_vars[cls_name] =\
        MyriadScalar(
            cls_name,
            MVoid,
            True,
            quals=["const"],
            init=ID("object + 1") if is_myriad_obj else None)
    return module_vars


def _initialize_obj_cls_structs(supercls: _MyriadObjectBase,
                                myriad_obj_vars: OrderedDict,
                                myriad_cls_vars: OrderedDict):
    """ Initializes object and class structs """
    if supercls is not _MyriadObjectBase:
        myriad_obj_vars["_"] = supercls.obj_struct("_", quals=["const"])
        myriad_cls_vars["_"] = supercls.cls_struct("_", quals=["const"])
    else:
        # Setup MyriadObject struct variables
        myriad_obj_vars["mclass"] = _gen_mclass_ptr_scalar("mclass")
        # Setup MyriadObjectClass struct variables
        tmp = MyriadScalar("_", MVoid, quals=["const"])
        tmp.type_decl = TypeDecl(declname="_", quals=[],
                                 type=Struct("MyriadObject", None))
        tmp.decl = Decl(name="_",
                        quals=["const"], storage=[],
                        funcspec=[], type=tmp.type_decl,
                        init=None, bitsize=None)
        myriad_cls_vars["_"] = tmp
        myriad_cls_vars["super"] = _gen_mclass_ptr_scalar("super")
        myriad_cls_vars["device_class"] =\
            _gen_mclass_ptr_scalar("device_class")
        myriad_cls_vars["size"] = MyriadScalar("size", MSizeT)


def _gen_mclass_ptr_scalar(ident: str):
    """ Quick & dirty way of hard-coding MyriadObjectClass struct pointers """
    tmp = MyriadScalar(ident,
                       MVoid,
                       True,
                       quals=["const"])
    tmp.type_decl = TypeDecl(declname=ident,
                             quals=[],
                             type=Struct("MyriadObjectClass", None))
    tmp.ptr_decl = PtrDecl(quals=[],
                           type=tmp.type_decl)
    tmp.decl = Decl(name=ident,
                    quals=["const"],
                    storage=[],
                    funcspec=[],
                    type=tmp.ptr_decl,
                    init=None,
                    bitsize=None)
    return tmp


def _gen_init_fun(namespace: OrderedDict, supercls: _MyriadObjectBase) -> str:
    """ Generates the init* function for modules """
    if supercls is _MyriadObjectBase:
        template = MakoTemplate(MYRIADOBJECT_INIT_FUN_TEMPLATE, namespace)
        LOG.debug("Rendering init function for MyriadObject")
        template.render()
        return template.buffer
    # Make temporary dictionary since we need to add an extra value
    tmp_dict = {"super_obj": supercls.obj_name, "super_cls": supercls.cls_name}
    tmp_dict.update(namespace)
    template = MakoTemplate(INIT_OB_FUN_TEMPLATE, tmp_dict)
    LOG.debug("Rendering init function for %s", namespace["obj_name"])
    template.render()
    return template.buffer


class MyriadMetaclass(type):
    """
    TODO: Documentation for MyriadMetaclass
    """

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
        myriad_cls_vars = OrderedDict()
        myriad_obj_vars = OrderedDict()
        myriad_methods = OrderedDict()

        # Setup object with implicit superclass to start of struct definition
        _initialize_obj_cls_structs(supercls, myriad_obj_vars, myriad_cls_vars)

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
                                                            myriad_methods,
                                                            myriad_cls_vars)

        # Add #include's from system libraries, local files, and CUDA headers
        namespace["local_includes"], namespace["lib_includes"] =\
            _generate_includes(supercls)
        namespace["cuda_local_includes"], namespace["cuda_lib_includes"] =\
            _generate_cuda_includes(supercls)

        # Create myriad class struct
        namespace["cls_struct"] = MyriadStructType(namespace["cls_name"],
                                                   myriad_cls_vars)

        # Add other objects to namespace
        namespace["myriad_methods"] = myriad_methods
        namespace["myriad_obj_vars"] = myriad_obj_vars
        namespace["myriad_cls_vars"] = myriad_cls_vars

        # Fill in missing methods (ctor/etc.)
        supercls._fill_in_base_methods(namespace, myriad_methods)

        # Initialize module variables
        namespace["myriad_module_vars"] = _init_module_vars(
            namespace["obj_name"],
            namespace["cls_name"],
            namespace)

        # Initialize module functions
        namespace["init_fun"] = _gen_init_fun(namespace, supercls)

        # Write templates now that we have full information
        LOG.debug("Creating templates for class %s", name)
        namespace = _template_creator_helper(namespace)

        # Finally, delete function from namespace
        for method_id in myriad_methods.keys():
            if method_id in namespace:
                del namespace[method_id]

        # Generate internal module representation
        namespace["__init__"] = MyriadMetaclass.myriad_init
        namespace["__setattr__"] = MyriadMetaclass.myriad_set_attr
        return type.__new__(mcs, name, (supercls,), dict(namespace))

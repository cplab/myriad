"""
.. module:: myriad_module
   :platform: Linux
   :synopsis: Provides abstraction layer for creating inheritable modules.

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

import copy

from collections import OrderedDict
from inspect import signature, Parameter

from myriad_utils import enforce_annotations
from myriad_mako_wrapper import MakoFileTemplate, MakoTemplate

from myriad_types import MyriadScalar, MyriadFunction, MyriadStructType
from myriad_types import MVoid

from ast_function_assembler import pyfunbody_to_cbody

HEADER_FILE_TEMPLATE = open("template/header_file.mako", 'r').read()

CUH_FILE_TEMPLATE = open("template/cuda_header_file.mako", 'r').read()

C_FILE_TEMPLATE = open("template/c_file.mako", 'r').read()

# TODO: Finish PYC_COMP_FILE_TEMPLATE
PYC_COMP_FILE_TEMPLATE = open("template/pyc_file.mako", 'r').read()


class MyriadMethod(object):
    """
    Generic class for abstracting methods into 3 core components:

    1. Delegator - API entry point for public method calls
    2. Super Delegator - API entry point for subclasses
    3. Instance Function(s) - 'Internal' method definitions for each class

    It also contains internal templates for generating delegators and super
    delegators from scratch.
    """

    DELG_TEMPLATE = open("template/delegator_func.mako", 'r').read()

    SUPER_DELG_TEMPLATE = \
        open("template/super_delegator_func.mako", 'r').read()

    @enforce_annotations
    def __init__(self,
                 m_fxn: MyriadFunction,
                 instance_methods: dict=None,
                 inherited: bool=False):
        """
        Initializes a method from a function.

        The point of this class is to automatically create delegators for a
        method. This makes inheritance of methods easier since the delegators
        are not implemented by the subclass, only the instance methods are
        overwritten.

        Note that inherited methods do not create delegators, only instance
        methods.

        :param MyriadFunction m_fxn: Method's prototypical delegator function.
        :param dict instance_methods: Mapping of object/class names to methods.
        :param bool inherited: Flag for denoting this method is overloaded.
        """

        #: Flag is true if this method overrides a previously-defined method.
        self.inherited = inherited

        # Need to ensure this function has a typedef
        m_fxn.gen_typedef()

        #: Local storage for delegator function.
        self.delegator = m_fxn

        #: Mapping of object/class names to instance methods for each.
        self.instance_methods = {}

        # Initialize (default: None) instance method(s)
        # If we are given a string, assume this is the instance method body
        # and auto-generate the MyriadFunction wrapper.
        for obj_name, i_method in instance_methods.items():
            if type(i_method) is str:
                self.gen_instance_method_from_str(obj_name, i_method)
            else:
                raise NotImplementedError("Non-string instance methods.")

        # Create super delegator
        super_args = copy.copy(m_fxn.args_list)
        super_class_arg = MyriadScalar("_class",
                                       MVoid,
                                       True,
                                       ["const"])
        tmp_arg_indx = len(super_args) + 1
        super_args[tmp_arg_indx] = super_class_arg
        super_args.move_to_end(tmp_arg_indx, last=False)
        _delg = MyriadFunction("super_" + m_fxn.ident,
                               super_args,
                               m_fxn.ret_var)
        #: Automatically-generated super delegator for this method
        self.super_delegator = _delg
        #: Template for this method's delegator
        self.delg_template = MakoTemplate(self.DELG_TEMPLATE, vars(self))
        #: Template for this method's super delegator
        self.super_delg_template = MakoTemplate(self.SUPER_DELG_TEMPLATE,
                                                vars(self))

        # TODO: Implement instance method template
        #: Template for method's instance methods.
        self.instance_method_template = None

    def gen_instance_method_from_str(self,
                                     m_name: str,
                                     method_body: str):
        """
        Automatically generate a MyriadFunction wrapper for a method body.

        :param str m_name: Name to prepend to the instance method identifier.
        :param str method_body: String template to use as the method body.
        """
        _tmp_f = MyriadFunction(m_name + '_' + self.delegator.ident,
                                args_list=self.delegator.args_list,
                                ret_var=self.delegator.ret_var,
                                storage=['static'],
                                fun_def=method_body)
        self.instance_methods[m_name] = _tmp_f


class MyriadModule(object):
    """
    Represents an independent Myriad module (e.g. MyriadObject).

    Automatically generates the necessary C-level structs, variables, and
    function declarations based on a combination of novel definitions and
    an inherited "supermodule."

    Note that this class exists somewhat in parallel to the elsewhere-declared
    :class:`MyriadObject._MyriadObject`, due to the hard-coded contents inside
    the latter. The proper pattern for MyriadObject 'subclasses' is to
    explicitly inherit this class, but pass :class:`MyriadObject` as the
    supermodule parameter to __init__.
    """

    DEFAULT_LIB_INCLUDES = {"stdlib.h", "stdio.h", "assert.h", "string.h",
                            "stddef.h", "stdarg.h", "stdint.h"}

    DEFAULT_CUDA_INCLUDES = {"cuda_runtime.h", "cuda_runtime_api.h"}

    CLS_CTOR_TEMPLATE = open("templates/class_ctor_template.mako", 'r').read()

    CLS_CUDAFY_TEMPLATE = \
        open("templates/class_cudafy_template.mako", 'r').read()

    @enforce_annotations
    def __init__(self,
                 supermodule,
                 obj_name: str,
                 cls_name: str=None,
                 obj_vars: OrderedDict=None,
                 methods: OrderedDict=None,  # Looks like {str:function}
                 cuda: bool=False):
        """
        Initializes module namespace, structures, and templates.

        Methods is canonically supplied with a mapping {str: function} where
        function is an actual Python function. Previous versions utilized
        MyriadFunction but this behaviour is deprecated.

        :param MyriadModule supermodule: Myriad module this instance inherits.
        :param str obj_name: Name of the C object this module wraps.
        :param str cls_name: Name of the C class this module wraps.
        :param OrderedDict obj_vars: Mapping of object state variables by name.
        :param OrderedDict methods: Methods as a name:function map.
        :param bool cuda: CUDA support status for this module.
        """

        #: Indicates CUDA support status
        self.cuda = cuda

        #: Internal name for object
        self.obj_name = obj_name
        #: Internal name for class, by default ObjectNameClass
        self.cls_name = None
        if cls_name is None:
            self.cls_name = obj_name + "Class"
        else:
            self.cls_name = cls_name

        # Setup object with implicit superclass to start of struct definition
        if obj_vars is not None:
            _arg_indx = len(obj_vars) + 1
            obj_vars[_arg_indx] = supermodule.obj_struct("_", quals=["const"])
            obj_vars.move_to_end(_arg_indx, last=False)
        else:
            obj_vars = OrderedDict()

        #: MyriadStruct definition representing object state
        self.obj_struct = MyriadStructType(self.obj_name, obj_vars)

        # Pre-processing step: need to re-form signatures of methods.
        # We have to do this because method signatures are incorrect:
        # since they lack the "self" parameter it means that nothing downstream
        # from this point in the toolchain will be able to properly deal
        # with the necessary argument. Because there is somewhat of a circular
        # dependency in that the struct necessary for the annotation doesn't
        # yet exist, we have to do it at the first place we have that struct
        # declaration available. Using it we can create a scalar to a pointer
        # to the class' object struct and use that as the `self` annotation.
        #
        # However, re-writing the signature is trivial, but it cannot be forced
        # back into the original method, since inspect.Signature is immutable.
        # This requires us to create a copy of the function signature and
        # modify it such that it provides the correct type annotations.
        #
        # We then use what we have to auto-generate the MyriadFunctions
        # using a special class method. Ironically the class method, in order
        # to be generic, will take the full signature and deconstruct it in
        # order to generate the MyriadFunction. So we are doing a little bit of
        # extra work here, but because it keeps the interface generic it's ok.

        # Has to be a positional argument; will re-insert it at the start
        tmp_self = Parameter("self",
                             Parameter.POSITIONAL_OR_KEYWORD,
                             annotation=self.obj_struct("self", ptr=True))
        # For each method, we must convert it a MyriadFunction
        for m_ident, method in methods.items():
            # Check if conversion is unneeded/has already happened
            if type(method) is MyriadFunction:
                # print(m_ident + " is already a MyriadFunction, skipping...")
                continue
            # We have to bootstrap the methods dictionary so that the C code
            # can know whether method/function calls are legal or not.
            # TODO: Fix fbody
            fbody = pyfunbody_to_cbody(method,
                                       c_methods=methods,
                                       struct_members=self.obj_struct.members)
            sig = signature(method)  # Get current signature
            param_odict = sig.parameters.copy()
            param_odict["self"] = tmp_self
            param_odict.move_to_end("self", last=False)  # Pre-pend `self`
            # Generate new signature
            new_sig = sig.replace(parameters=list(param_odict.values()))
            # Generate new function
            methods[m_ident] = MyriadFunction.from_method_signature(m_ident,
                                                                    new_sig,
                                                                    fbody)

        # Set new methods and inherit old ones
        #
        # The idea here is to preserve all methods from the supermodule,
        # so that our subclasses can overwrite our implementations even though
        # the methods originated in a class farther up the tree (the best
        # example of this is myriad_ctor, which nearly everyone overwrites).
        #
        # In order to do this, we need to copy all methods from our superclass,
        # taking care to set the "inherited" flag to True for each of them.
        # Resetting the instance methods dict is crucial to prevent duplicate
        # declarations. After reset but before adding to our own methods,
        # we inject our overrides (if any). It is also important that new
        # methods are added to the dictionary last and in the same order
        # as provided in the arguments, in order to ensure proper ordering
        # in the class struct.

        #: Ordered dictionary of this module's method names
        self.methods = OrderedDict()

        # Import super methods
        for m_ident, method in copy.copy(supermodule.methods).items():
            new_instance_methods = {}

            # If method is going to be overriden, add instance method provided
            if m_ident in methods:
                # This assumes `methods` at this point is {str:MyriadFunction}
                new_instance_methods[self.obj_name] = methods[m_ident].fun_def
                del methods[m_ident]
            new_method = MyriadMethod(method.delegator,
                                      new_instance_methods,
                                      True)
            self.methods[m_ident] = new_method

        # Add new methods.
        #
        # Currently using str:MyriadFunction because it is cleanest method,
        # since this means parity with supermodule's method lists, however
        # this doesn't work when we need both object and class to have the
        # same method. However, outside of MyriadObject/Class (which is a
        # special case) I don't think ^that will ever be needed by a user.
        #
        # We only use str:MyriadFunction for incoming methods. Why? Because the
        # delegator will be automatically generated. This means that the
        # function we are passed is actually the method in and of itself:
        # it is a declaration and a function body. The declaration is SHARED
        # between the delegator generated by MyriadMethod AND the instance
        # method declaration generator of the same. Since the user only writes
        # one method anyways (the instance method), we can safely extrapolate
        # from its function annotations what its delegators will look like,
        # since they must share the same function signature and typedef. The
        # body of the delegator already uses an internal template, it is the
        # body of the instance method that is passed to us.
        for method_ident, fxn in methods.items():
            tmp_dict = {self.obj_name: fxn.fun_def}
            self.methods[method_ident] = MyriadMethod(fxn, tmp_dict)

        # Initialize class object

        # Struct variables are function pointers for methods
        cls_vars = OrderedDict()
        cls_vars["_"] = supermodule.cls_struct("_", quals=["const"])

        for mtd in [mtd for mtd in self.methods.values() if not mtd.inherited]:
            m_scal = MyriadScalar("my_" + mtd.delegator.fun_typedef.name,
                                  mtd.delegator.base_type)
            cls_vars[m_scal.ident] = m_scal

        #: MyriadStruct definition representing object state
        self.cls_struct = MyriadStructType(self.cls_name, cls_vars)

        # Initialize Class constructor if new methods are created
        # Checks if superclass instance ctor exists, then delete it
        if len(self.methods) > len(supermodule.methods):
            _cls_tmplt = MakoTemplate(self.CLS_CTOR_TEMPLATE, vars(self))
            _cls_tmplt.render()
            _cls_ctor_mtd = self.methods["myriad_ctor"]
            _cls_ctor_mtd.gen_instance_method_from_str(self.cls_name,
                                                       _cls_tmplt.buffer)

        # TODO: Add init* function
        #: Mapping of global module functions
        self.functions = OrderedDict()

        #: Dictionary of module global variables
        self.module_vars = OrderedDict()
        self.module_vars[self.obj_name] = MyriadScalar(self.obj_name,
                                                       MVoid,
                                                       True,
                                                       quals=["const"])
        self.module_vars[self.cls_name] = MyriadScalar(self.cls_name,
                                                       MVoid,
                                                       True,
                                                       quals=["const"])

        # Initialize standard library imports, by default with fail-safes
        self.lib_includes = MyriadModule.DEFAULT_LIB_INCLUDES

        # TODO: Initialize local header imports
        #: Set of local *.H/*.CUH include headers
        self.local_includes = set()
        self.local_includes.add(supermodule.obj_name + ".h")
        self.local_includes.update(supermodule.local_includes)

        # Initialize C file templates
        #: C header file template
        self.header_template = self.create_file_template(".h",
                                                         HEADER_FILE_TEMPLATE)
        #: C file template
        self.c_file_template = self.create_file_template(".c",
                                                         C_FILE_TEMPLATE)
        #: CUDA header file template
        self.cuh_file_template = self.create_file_template(".cuh",
                                                           CUH_FILE_TEMPLATE)
        # TODO: CUDA implementation file template
        # TODO: CPython implementation file template

    def create_file_template(self,
                             suffix: str,
                             template_str: str,
                             context_dict: dict=None):
        """
        Initializes internal Mako C/CUDA templates.

        :param str suffix: Suffix of the template file.
        :param str template_str: String object representing the template.
        :param dict context_dict: Dictionary passed to Mako template context.

        :return: File template ready for rendering.
        :rtype: MakoFileTemplate
        """
        if context_dict is None:
            context_dict = vars(self)
        return MakoFileTemplate(self.obj_name + suffix,
                                template_str,
                                context_dict)


def main():
    pass

if __name__ == "__main__":
    main()

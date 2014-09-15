#!/usr/bin/python3
"""
TODO: Docstring
"""

import copy

from collections import OrderedDict

from myriad_utils import enforce_annotations, TypeEnforcer
from myriad_mako_wrapper import MakoFileTemplate, MakoTemplate

from myriad_types import MyriadScalar, MyriadFunction, MyriadStructType
from myriad_types import MVoid, MyriadFunType


HEADER_FILE_TEMPLATE = """

## Python imports as a module-level block
<%!
    import myriad_types
%>

## Add include guards
<% include_guard = obj_name.upper() + "_H" %>
#ifndef ${include_guard}
#define ${include_guard}

## Add library includes
% for lib in lib_includes:
#include <${lib}>
% endfor

## Add local includes
% for lib in local_includes:
#include "${lib}"
% endfor

## Declare typedefs
% for method in methods:
${method.delegator.stringify_typedef()};
% endfor

## Struct forward declarations
struct ${cls_name};
struct ${obj_name};

## Module variables
% for m_var in module_vars:
extern ${m_var.stringify_decl()};
% endfor

// Top-level functions

extern int initCUDAObjects();

extern void* myriad_new(const void* _class, ...);

extern const void* myriad_class_of(const void* _self);

extern size_t myriad_size_of(const void* self);

extern int myriad_is_a(const void* _self, const struct MyriadClass* m_class);

extern int myriad_is_of(const void* _self, const struct MyriadClass* m_class);

extern const void* myriad_super(const void* _self);

// Methods

% for method in methods:
extern ${method.delegator.stringify_decl()};
% endfor

// Super delegators

% for method in methods:
extern ${method.super_delegator.stringify_decl()};
% endfor

// Class/Object structs

${obj_struct.stringify_decl()}
${cls_struct.stringify_decl()}

#endif

"""

# pylint: disable=R0902
# pylint: disable=R0903
class MyriadMethod(object):
    """
    Generic class for abstracting methods into 3 core components:

    1) Delegator - API entry point for public method calls
    2) Super Delegator - API entry point for subclasses
    3) Instance Function - Internal definition of method on a per-class basis.
    """

    DELG_TEMPLATE = """
<%
    fun_args = ','.join([arg.ident for arg in delegator.args_list.values()])
%>

${delegator.stringify_decl()}
{
    const struct MyriadClass* m_class = (const struct MyriadClass*) myriad_class_of(${delegator.args_list[0].ident});

    assert(m_class->${delegator.fun_typedef.name});

    % if delegator.ret_var.base_type is MVoid and not delegator.ret_var.base_type.ptr:
    m_class->my_${delegator.fun_typedef.name}(${fun_args});
    return;
    % else:
    return m_class->my_${delegator.fun_typedef.name}(${fun_args});
    % endif
}
    """

    SUPER_DELG_TEMPLATE = """
<%
    fun_args = ','.join([arg.ident for arg in super_delegator.args_list.values()])
%>
${supert_delegator.stringify_decl()}
{
    const struct MyriadClass* superclass = (const struct MyriadClass*) myriad_super(${super_delegator.args_list[0].ident});

    assert(superclass->${delegator.fun_typedef.name});

    % if delegator.ret_var.base_type is MVoid and not delegator.ret_var.base_type.ptr:
    superclass->my_${delegator.fun_typedef.name}(${fun_args});
    return;
    % else:
    return superclass->my_${delegator.fun_typedef.name}(${fun_args});
    % endif
}
    """

    @enforce_annotations
    def __init__(self,
                 m_fxn: MyriadFunction,
                 instance_method=None,
                 obj_name: str=None):
        """
        Initializes a method from a function.

        The point of this class is to automatically create delegators for a
        method. This makes inheritance of methods easier since the delegators
        are not implemented by the subclass, only the instance methods are
        overwritten.
        """

        # Need to ensure this function has a typedef
        m_fxn.gen_typedef()
        self.delegator = m_fxn

        # Initialize (default: None) instance method
        self.instance_method = instance_method
        # If we are given a string, assume this is the instance method body
        # and auto-generate the MyriadFunction wrapper
        if type(self.instance_method) is str:
            if obj_name is None:
                raise ValueError("Must provide instance method object name.")
            self.gen_instance_method_from_str(self.instance_method, obj_name)

        # Create super delegator
        super_args = copy.copy(m_fxn.args_list)
        super_class_arg = MyriadScalar("_class",
                                       MVoid,
                                       True,
                                       ["const"])
        tmp_arg_indx = len(super_args)+1
        super_args[tmp_arg_indx] = super_class_arg
        super_args.move_to_end(tmp_arg_indx, last=False)
        _delg = MyriadFunction("super_" + m_fxn.ident,
                               super_args,
                               m_fxn.ret_var,
                               MyriadFunType.m_delg)
        self.super_delegator = _delg
        self.delg_template = MakoTemplate(self.DELG_TEMPLATE,
                                          vars(self))
        self.super_delg_template = MakoTemplate(self.SUPER_DELG_TEMPLATE,
                                                vars(self))
        # TODO: Implement instance method template
        self.instance_method_template = None

    def gen_instance_method_from_str(self,
                                     method_body: str,
                                     obj_name: str) -> MyriadFunction:
        """
        Automatically generate a MyriadFunction wrapper for a method body.
        """
        _tmp_f = MyriadFunction(obj_name + '_' + self.delegator.ident,
                                args_list=self.delegator.args_list,
                                ret_var=self.delegator.ret_var,
                                fun_type=MyriadFunType.m_module,
                                fun_def=method_body)
        self.instance_method = _tmp_f


# pylint: disable=R0902
class MyriadModule(object, metaclass=TypeEnforcer):
    """
    Represents an independent Myriad module (e.g. MyriadObject)
    """

    DEFAULT_LIB_INCLUDES = {"stdlib.h", "stdio.h", "assert.h",
                            "stddef.h", "stdarg.h", "stdint.h"}

    DEFAULT_CUDA_INCLUDES = {"cuda_runtime.h", "cuda_runtime_api.h"}

    @enforce_annotations
    def __init__(self,
                 supermodule,
                 obj_name: str,
                 cls_name: str=None,
                 obj_vars: OrderedDict=None,
                 methods: set=None,
                 cuda: bool=False):
        """Initializes a module"""

        # Set CUDA support status
        self.cuda = cuda

        # Set internal names for classes
        self.obj_name = obj_name
        if cls_name is None:
            self.cls_name = obj_name + "Class"
        else:
            self.cls_name = cls_name

        # methods = delegator, super delegator, instance
        # We assume (for now) that the instance methods are uninitialized
        self.methods = set()
        methods = set() if methods is None else methods
        for method in methods:
            self.methods.add(MyriadMethod(method))

        # Initialize class object and object class

        # Add implicit superclass to start of struct definition
        if obj_vars is not None:
            _arg_indx = len(obj_vars)+1
            obj_vars[_arg_indx] = supermodule.cls_struct("_", quals=["const"])
            obj_vars.move_to_end(_arg_indx, last=False)
        else:
            obj_vars = OrderedDict()
        self.obj_struct = MyriadStructType(self.obj_name, obj_vars)

        # Initialize class variables, i.e. function pointers for methods
        cls_vars = OrderedDict()
        cls_vars[0] = supermodule.cls_struct("_", quals=["const"])

        for indx, method in enumerate(self.methods):
            m_scal = MyriadScalar("my_" + method.delegator.fun_typedef.name,
                                  method.delegator.base_type)
            cls_vars[indx+1] = m_scal

        self.cls_vars = cls_vars
        self.cls_struct = MyriadStructType(self.cls_name, self.cls_vars)

        # TODO: Dictionaries or sets?
        self.functions = set()

        # Initialize module global variables
        self.module_vars = set()
        v_obj = MyriadScalar(self.obj_name,
                             MVoid,
                             True,
                             quals=["const"])
        self.module_vars.add(v_obj)
        v_cls = MyriadScalar(self.cls_name,
                             MVoid,
                             True,
                             quals=["const"])
        self.module_vars.add(v_cls)

        # Initialize standard library imports, by default with fail-safes
        self.lib_includes = MyriadModule.DEFAULT_LIB_INCLUDES

        # TODO: Initialize local header imports
        self.local_includes = set()

        # Initialize C header template
        self.header_template = None
        self.initialize_header_template()

    def register_module_function(self,
                                 function: MyriadFunction,
                                 strict: bool=False,
                                 override: bool=False):
        """
        Registers a global function in the module.

        Note: strict and override are mutually exclusive.

        Keyword arguments:
        method -- method to be registered
        strict -- if True, raises an error if a collision occurs when joining.
        override -- if True, overrides superclass methods.
        """
        if strict is True and override is True:
            raise ValueError("Flags strict and override cannot both be True.")

        # TODO: Make "override"/"strict" modes check for existance better.
        if function in self.functions:
            if strict:
                raise ValueError("Cannot add duplicate functions.")
            elif override:
                self.functions.discard(function)
        self.functions.add(function)

    def initialize_header_template(self, context_dict: dict=None):
        """ Initializes internal Mako template for C header file. """
        if context_dict is None:
            context_dict = vars(self)
        self.header_template = MakoFileTemplate(self.obj_name+".h",
                                                HEADER_FILE_TEMPLATE,
                                                context_dict)

    def render_header_template(self, printout: bool=False):
        """ Renders the header file template. """

        # Reset buffer if necessary
        if self.header_template.buffer is not '':
            self.header_template.reset_buffer()

        self.header_template.render()

        if printout:
            print(self.header_template.buffer)
        else:
            self.header_template.render_to_file()


def main():
    pass


if __name__ == "__main__":
    main()

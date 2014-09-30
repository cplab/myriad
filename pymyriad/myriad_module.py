#!/usr/bin/python3
"""
TODO: Docstring
"""

import copy

from collections import OrderedDict

from myriad_utils import enforce_annotations, TypeEnforcer
from myriad_mako_wrapper import MakoFileTemplate, MakoTemplate

from myriad_types import MyriadScalar, MyriadFunction, MyriadStructType
from myriad_types import MVoid


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
% for method in methods.values():
    % if not method.inherited:
${method.delegator.stringify_typedef()};
    % endif
% endfor

## Struct forward declarations
struct ${cls_name};
struct ${obj_name};

## Module variables
% for m_var in module_vars.values():
    % if type(m_var) is not str and 'static' not in m_var.decl.storage:
extern ${m_var.stringify_decl()};
    % endif
% endfor

## Top-level functions
% for fun in functions:
extern ${fun.stringify_decl()};
% endfor

## Method delegators
% for method in [m for m in methods.values() if not m.inherited]:

extern ${method.delegator.stringify_decl()};

extern ${method.super_delegator.stringify_decl()};

% endfor

## Class/Object structs
${obj_struct.stringify_decl()}
${cls_struct.stringify_decl()}

#endif
"""

C_FILE_TEMPLATE = """

## Python imports as a module-level block
<%!
    import myriad_types
%>

#include "myriad_debug.h"

## Add local includes
% for lib in local_includes:
#include "${lib}"
% endfor

#include "${obj_name}.h"

## Print methods forward declarations
% for method in methods.values():
    % for i_method in method.instance_methods.values():
${i_method.stringify_decl()};
    % endfor
% endfor

## Print top-level module variables
% for module_var in module_vars.values():
    % if type(module_var) is str:
${module_var}
    % else:
        % if module_var.init is not None:
${module_var.stringify_decl()} = ${module_var.init};
        % else:
${module_var.stringify_decl()};
        % endif
    % endif
% endfor

## Method definitions
% for method in methods.values():
    % for i_method in method.instance_methods.values():
${i_method.stringify_decl()}
{
    ${i_method.fun_def}
}
    % endfor

## Use this trick to force rendering before printing the buffer in one line
${method.delg_template.render() or method.delg_template.buffer}

${method.super_delg_template.render() or method.super_delg_template.buffer}

% endfor

## Top-level functions
% for fun in functions:
${fun.stringify_decl()}
{
    ${fun.fun_def}
}
% endfor
"""


# pylint: disable=R0902
# pylint: disable=R0903
class MyriadMethod(object):
    """
    Generic class for abstracting methods into 3 core components:

    1) Delegator - API entry point for public method calls
    2) Super Delegator - API entry point for subclasses
    3) Instance Function(s) - 'Internal' method definitions for each class
    """

    DELG_TEMPLATE = """
% if not inherited:
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
% endif
    """

    SUPER_DELG_TEMPLATE = """
% if not inherited:
<%
    fun_args = ','.join([arg.ident for arg in super_delegator.args_list.values()])
%>
${super_delegator.stringify_decl()}
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
% endif
    """

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
        """

        self.inherited = inherited

        # Need to ensure this function has a typedef
        m_fxn.gen_typedef()
        self.delegator = m_fxn

        # Initialize (default: None) instance method
        self.instance_methods = {}
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
        self.super_delegator = _delg
        self.delg_template = MakoTemplate(self.DELG_TEMPLATE, vars(self))
        self.super_delg_template = MakoTemplate(self.SUPER_DELG_TEMPLATE,
                                                vars(self))

        # TODO: Implement instance method template
        self.instance_method_template = None

    def gen_instance_method_from_str(self,
                                     obj_name: str,
                                     method_body: str):
        """
        Automatically generate a MyriadFunction wrapper for a method body.
        """
        _tmp_f = MyriadFunction(obj_name + '_' + self.delegator.ident,
                                args_list=self.delegator.args_list,
                                ret_var=self.delegator.ret_var,
                                storage=['static'],
                                fun_def=method_body)
        self.instance_methods[obj_name] = _tmp_f


# pylint: disable=R0902
class MyriadModule(object):
    """
    Represents an independent Myriad module (e.g. MyriadObject).
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
                 methods: OrderedDict=None,  # Looks like str:MyriadFunction
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

        # Set new methods and inherit old ones
        #
        # The idea here is to preserve all methods from the supermodule,
        # so that our subclasses can overwrite our implementations even though
        # the methods originated in a class farther up the tree (the best
        # example of this is myriad_ctor, which nearly everyone overwrites).
        #
        # In order to do this, we need to copy all methods from our superclass,
        # taking care to set the "inherited" flag to True for each of them.
        # Resetting the instance methods dict is crucial to prevent double-
        # -declarations. After reset but before adding to our own methods,
        # we inject our overrides (if any). It is also important that new
        # methods are added to the dictionary last and in the same order
        # as provided in the arguments, in order to ensure proper ordering
        # in the class struct.
        self.methods = OrderedDict()

        # Import super methods
        super_methods = copy.deepcopy(supermodule.methods)
        for m_ident, method in super_methods:
            method.inherited = True
            method.instance_methods = {}
            # If method is going to be overriden, add instance method provided
            if m_ident in methods:
                # This assumes `methods` is str:MyriadFunction
                method.instance_methods[m_ident] = methods[m_ident].fun_def
                del methods[m_ident]
            self.methods[m_ident] = method

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
        for method_ident, fxn in methods:
            tmp_dict = {self.obj_name: fxn.fun_def}
            self.methods[method_ident] = MyriadMethod(fxn, tmp_dict)

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

        for indx, method in enumerate(self.methods.values()):
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

        # Initialize C file template
        self.c_file_template = None
        self.initialize_c_file_template()

    def initialize_c_file_template(self, context_dict: dict=None):
        """ Initializes internal Mako template for C file. """
        if context_dict is None:
            context_dict = vars(self)
        self.c_file_template = MakoFileTemplate(self.obj_name+".c",
                                                C_FILE_TEMPLATE,
                                                context_dict)

    def initialize_header_template(self, context_dict: dict=None):
        """ Initializes internal Mako template for C header file. """
        if context_dict is None:
            context_dict = vars(self)
        self.header_template = MakoFileTemplate(self.obj_name+".h",
                                                HEADER_FILE_TEMPLATE,
                                                context_dict)


def main():
    pass


if __name__ == "__main__":
    main()

#!/usr/bin/python3
"""
TODO: Docstring
"""

from mako.template import Template
from mako.runtime import Context
from io import StringIO

from m_annotations import enforce_annotations

import myriad_types


HEADER_FILE_TEMPLATE = """

## Add include guards
<% include_guard = object_name.upper() + "_H" %>
#ifndef ${include_guard}
#define ${include_guard}

## Add library includes
% for lib in lib_includes:
#include <${lib}.h>
% endfor

## Add local includes
% for lib in local_includes:
#include "${lib}.h"
% endfor

## Declare typedefs
<%
for fun in functions:
    if fun.gen_typedef is not None:
        context.write(f.stringify_typedef())
%>


## Struct forward declarations
struct ${class_name};
struct ${object_name};

extern const void* MyriadObject;
extern const void* MyriadClass;

## Print methods forward declarations
<%
for fun in functions:
    if fun.is_global:
        context.write("extern "+f.stringify_decl()+";")
%>

extern int initCUDAObjects();

extern void* myriad_new(const void* _class, ...);

extern const void* myriad_class_of(const void* _self);

extern size_t myriad_size_of(const void* self);

extern int myriad_is_a(const void* _self, const struct MyriadClass* m_class);

extern int myriad_is_of(const void* _self, const struct MyriadClass* m_class);

// Delegators

extern void* myriad_ctor(void* _self, va_list* app);

extern int myriad_dtor(void* _self);

extern void* myriad_cudafy(void* _self, int clobber);

extern void myriad_decudafy(void* _self, void* cuda_self);

extern const void* myriad_super(const void* _self);

extern void* super_ctor(const void* _class, void* _self, va_list* app);

extern int super_dtor(const void* _class, void* _self);

extern void* super_cudafy(const void* _class, void* _self, int clobber);

extern void super_decudafy(const void* _class, void* _self, void* cuda_self);

struct MyriadObject
{
    const struct MyriadClass* m_class; //! Object's class/description
};

struct MyriadClass
{
    const struct MyriadObject _;
    const struct MyriadClass* super;
    const struct MyriadClass* device_class;
    size_t size;
    ctor_t my_ctor;
    dtor_t my_dtor;
    cudafy_t my_cudafy;
    de_cudafy_t my_decudafy;
};

#endif

"""


class MyriadModule(object):
    """
    Represents an independent Myriad module (e.g. MyriadObject)
    """

    FAILSAFE_INCLUDES = ["stdlib", "stdio", "assert",
                         "stddef", "stdarg", "stdint"]

    @enforce_annotations
    def __init__(self,
                 object_name: str,
                 class_name: str=None,
                 lib_includes: list=None):
        """Initializes a module"""

        self.object_name = object_name

        if class_name is None:
            self.class_name = object_name + "Class"
        else:
            self.class_name = class_name

        # TODO: Initialize functions/methods/delegators
        self.functions = []

        # Initialize module global variables
        self.module_vars = []

        # Initialize standard library imports, by default with fail-safes
        self.lib_includes = lib_includes
        if self.lib_includes is None:
            self.lib_includes = MyriadModule.FAILSAFE_INCLUDES

        # Initialize C header template
        self.header_template = None
        self.header_buffer = None
        self.header_ctx = None
        self.initialize_header_template()

    @enforce_annotations
    def initialize_header_template(self, context_dict: dict=None):
        """ Initializes internal Mako template for C header file """
        self.header_template = Template(HEADER_FILE_TEMPLATE)
        self.header_buffer = StringIO()

        if context_dict is None:
            self.header_ctx = Context(self.header_buffer, **vars(self))
        else:
            self.header_ctx = Context(self.header_buffer, **context_dict)

    @enforce_annotations
    def render_header_template(self, buf: StringIO=None, printout: bool=False):
        """ Renders the header file template """
        if buf is not None:
            self.header_buffer = buf

        self.header_template.render_context(self.header_ctx)

        if printout:
            print(self.header_buffer.getvalue())
        else:
            # TODO: What to do when not printing to stdout?
            pass

    def register_global_variable(self, var: myriad_types.MyriadScalar):
        pass

def main():
    """ Creates a template and renders it using a Mako context """
    m_object = MyriadModule("MyriadObject")
    m_object.render_header_template(printout=True)

if __name__ == "__main__":
    main()

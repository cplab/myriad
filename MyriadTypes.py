#!/usr/bin/python3
"""
typedef void (* voidf) ();

typedef void* (* ctor_t) (void* self, va_list* app);

typedef int (* dtor_t) (void* self);

typedef void* (* cudafy_t) (void* self, int clobber);

typedef void (* de_cudafy_t) (void* self, void* cuda_self);
"""
from mako.runtime import Context
from mako.template import Template
from io import StringIO
from m_annotations import enforce_annotations
from enum import Enum


class _MakoTemplate(object):

    @enforce_annotations
    def __init__(self,
                 template: str="",
                 buf: StringIO=None,
                 ctx: Context=None):
        """
        Initializes a (by default, empty) mako template, context, and buffer.

        Keyword arguments:
        template -- actual mako template string (default empty string)
        buf -- StringIO buffer for rendering (default empty buffer)
        ctx -- mako runtime context for rendering (default empty context)
        """
        self.template = Template() if template is None else Template(template)
        self.str_buffer = StringIO() if buf is None else buf
        self.ctx = Context(self.str_buffer, **{}) if ctx is None else ctx

    def render(self, filename=None):
        # TODO: Make this file/filename agnostic
        if (self.str_buffer is not "" and self.str_buffer is not None):
            self.str_buffer = StringIO()  # Refresh the buffer
        self.template.render_context(self.ctx)

    def __str__(self):
        self.render()
        return self.str_buffer


class MyriadCType(Enum):
    m_float = "float"
    m_double = "double"
    m_int = "int"
    m_uint = "unsigned int"
    m_void = "void"


class MyriadVariable(object):

    @enforce_annotations
    def __init__(self,
                 ident: str,
                 c_type: MyriadCType,
                 ptr: int=0):
        """
        Initializes a MyriadVariable with an indentifier and a native c type.

        A level of pointer indirection may also be provided.

        Keyword Arguments:
        ident -- Identifier (for use in source-to-source translation)
        c_type -- Underlying C type
        ptr -- Optional level of pointer indirection (default: 0)
        """
        self.ident = ident
        self.c_type = c_type
        self.ptr = ptr
        self._ptr_repr = "".join("*" for i in range(self.ptr))

        # Initialize internal templates
        self._decl_template = _MakoTemplate(template="""
        ${c_type} ${_ptr_repr} ${ident};
        """)


class MyriadArray(MyriadVariable):

    @enforce_annotations
    def __init__(self,
                 ident: str,
                 base_type: MyriadCType,
                 dim_len: list,
                 dynamic: bool=True):
        """
        Initializes a Myriad Array variable.

        Keyword Arguments
        ident -- identifier for the array variable
        base_type -- underlying c type
        dim_len -- list of lengths of each dimension (0 if dynamic)
        dynamic -- boolean flag for dynamic vs static initialization
        """
        assert(dim_len is not None and len(dim_len) != 0)
        assert(type(i) is int and i >= 0 for i in dim_len)
        super().__init__(ident=ident, c_type=base_type, ptr=len(dim_len))
        self.dim_len = dim_len
        self.dynamic = dynamic



def main():
    pass

if __name__ == "__main__":
    main()

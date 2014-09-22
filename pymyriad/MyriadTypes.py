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
        self.template = Template(template)
        self.str_buffer = StringIO() if buf is None else buf
        self.ctx = Context(self.str_buffer, **{}) if ctx is None else ctx

    @enforce_annotations
    def set_context(self, m_dict: dict={}):
        self.str_buffer = StringIO()
        self.ctx = Context(self.str_buffer, **m_dict)

    def render(self):
        if (self.str_buffer.getvalue() is not "" or self.str_buffer is None):
            self.str_buffer = StringIO()  # Refresh the buffer
        self.template.render_context(self.ctx)

    def __str__(self):
        self.render()
        return self.str_buffer.getvalue()


class MyriadCType(Enum):
    m_float = "float"
    m_double = "double"
    m_int = "int"
    m_uint = "unsigned int"
    m_void = "void"
    m_struct = "struct"


class MyriadVariable(object):

    @enforce_annotations
    def __init__(self,
                 ident: str,
                 c_type: MyriadCType,
                 ptr: int=0,
                 decl_template: str=None):
        """
        Initializes a MyriadVariable with an indentifier and a native c type.

        A level of pointer indirection may also be provided.

        Keyword Arguments:
        ident -- Identifier (for use in source-to-source translation)
        c_type -- Underlying C type
        ptr -- Optional level of pointer indirection (default: 0)
        """
        if c_type is MyriadCType.m_void and ptr == 0:
            raise TypeError("Can't create a void variable")

        self.ident = ident
        self.c_type = c_type
        self.c_type_repr = c_type.value
        self.ptr = ptr
        self.ptr_repr = "".join("*" for i in range(self.ptr))

        # Initialize internal templates

        # Stand-alone declaration
        if decl_template is None:
            self.decl_template = _MakoTemplate(template="""
            ${c_type_repr} ${ptr_repr} ${ident}""")
        else:
            self.decl_template = _MakoTemplate(template=decl_template)

    def declaration(self):
        self.decl_template.set_context(vars(self))
        return str(self.decl_template)


class MyriadVector(MyriadVariable):

    @enforce_annotations
    def __init__(self,
                 ident: str,
                 base_type: MyriadCType,
                 m_len: int,
                 dynamic: bool=True):
        """
        Initializes a Myriad vector variable.

        Keyword Arguments
        ident -- identifier for the array variable
        base_type -- underlying c type
        m_len -- length of the vector
        dynamic -- boolean flag for dynamic vs static initialization
        """
        assert(m_len > 0)

        # Only use */**/etc. if dynamically initialized
        super().__init__(ident=ident,
                         c_type=base_type,
                         ptr=1 if dynamic else 0)

        self.m_len = m_len
        self.dynamic = dynamic

        # Interal templates
        self.init_template = _MakoTemplate("""
        % if not dynamic:
        ${c_type_repr} ${ptr_repr} ${ident} [${m_len}]
        % else:
        ${c_type_repr} ${ptr_repr} ${ident} =
        (${c_type_repr} ${ptr_repr}) calloc(${m_len}, sizeof(${c_type_repr}))
        % endif
        """)
        self.init_template.set_context(vars(self))


class MyriadFunction(object):

    @enforce_annotations
    def __init__(self,
                 ident: str,
                 args_list: list,
                 ret_var: MyriadVariable=None):
        self.ident = ident
        self.args_list = args_list
        self.ret_var = ret_var

        if self.ret_var is None:
            self.ret_type = MyriadCType.m_void
        else:
            self.ret_type = self.ret_var.c_type

        # Internal templates
        self._decl_template = _MakoTemplate(template="""
        ${ret_type} ${ident}
        (
        % for m_arg in m_args_list:
        ${m_arg},
        % endfor
        )
        """)
        self.fxn_body_template = _MakoTemplate(template="")


def main():
    a = MyriadVariable("a", MyriadCType.m_int)
    print(a.declaration())
    v = MyriadVector("v", MyriadCType.m_double, 5)
    print(v.declaration())
    v.init_template.set_context(vars(v))
    print(v.init_template)

if __name__ == "__main__":
    main()

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
from functools import wraps
from inspect import getcallargs


def enforce_annotations(f):
    """
    Function annotation to enforce function argument and return types.
    """
    @wraps(f)
    def _wrapper(*args, **kwargs):
        # Check each annotated argument type for type correctness
        for arg, val in getcallargs(f, *args, **kwargs).items():
            if arg in f.__annotations__:
                templ = f.__annotations__[arg]
                msg_args = {'arg': arg, 'f': f, 't1': type(val), 't2': templ}
                msg = """Argument mismatch in call to {f}:
                \'{arg}\' is of type {t1}, expected type {t2}"""
                if val is not None and not issubclass(val.__class__, templ):
                    raise TypeError(msg.format(**msg_args))
        # Call wrapped function and get return value
        return_val = f(*args, **kwargs)
        if 'return' in f.__annotations__:
            templ = f.__annotations__['return']
            msg_args = {'f': f, 't1': type(return_val), 't2': templ}
            msg = """Return type mismatch in call to {f}:
            Call return type {t1} does not match expected type {t2}"""
            if (return_val is not None
                    and not issubclass(return_val.__class__, templ)):
                raise TypeError(msg.format(**msg_args))
        return return_val
    return _wrapper


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
        self.template = template if template is not None else Template()
        self.str_buffer = buf if buf is not None else StringIO()
        self.ctx = ctx if ctx is not None else Context(self.str_buffer, **{})

    def render(self, filename=None):
        # TODO: Make this file/filename agnostic
        if (self.str_buffer is not "" and self.str_buffer is not None):
            self.str_buffer = StringIO()  # Refresh the buffer
        self.template.render_context(self.mako_context)

    def __str__(self):
        self.render()
        return self.str_buffer


class MyriadBasicType(object):

    def __init__(self):
        pass


class MyriadMethod(object):

    @enforce_annotations
    def __init__(self,
                 c_name: str=None,
                 c_ret_type: str="void",
                 c_args: dict={}):
        self.c_name = c_name if c_name is not None else str(self.__class__)
        self.c_ret_type = c_ret_type
        self.c_args = c_args  # TODO: Use regimented paramater type list

    @classmethod
    def from_method(cls, method):
        raise NotImplementedError()


class MyriadModule(object):

    def __init__(self):
        self.lib_includes = ["stdio.h", "stdlib.h", "stddef.h"]
        self.local_includes = ["assert.h"]
        self.method_types = []


class MyriadObject(object):

    def __init__(self, m_super_class_name=None, template=None):

        if m_super_class_name is None:
            self.m_super_class_name = str(self.__class__.__name__)
        else:
            self.m_super_class_name = m_super_class_name

        if template is None:
            self.mako_template = _MakoTemplate(u"""
            struct MyriadObject
            {
                const struct MyriadClass* m_class;
            }
            """)
        else:
            self.mako_template = template


class MyriadClass(MyriadObject):

    def __init__(self):
        self.mako_template = _MakoTemplate(u"""
        struct ${classname}
        {
            const struct ${m_super_class_name} _;
            % for m_methods in m_attributes
            ${m_methods};
            % endfor
        };
        """)


def main():
    pass

if __name__ == "__main__":
    main()

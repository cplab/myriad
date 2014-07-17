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
    Function annotation to enforce argument and return types.
    """
    @wraps(f)
    def _wrapper(*args, **kwargs):
        for arg, val in getcallargs(f, *args, **kwargs).items():
            if arg in f.__annotations__:
                templ = f.__annotations__[arg]
                msg = "Argument \'{arg}\' of type {t1} to {f} doesn't match annotation type {t2}"
                if (val is not None and not issubclass(val.__class__, templ)):
                    raise ValueError(msg.format(arg=arg, f=f, t1=type(val), t2=templ))
        return_val = f(*args, **kwargs)
        if 'return' in f.__annotations__:
            templ = f.__annotations__['return']
            msg = "Return value of {f} does not match annotation type {t}"
            if (val is not None
                    and not issubclass(val.__class__, templ.__class__)):
                raise ValueError(msg.format(arg=arg, f=f, t=templ))
        return return_val
    return _wrapper


class _MakoTemplate(object):

    @enforce_annotations
    def __init__(self,
                 template: str="",
                 buf: StringIO=None,
                 ctx: Context=None):
        self.template = template if template is not None else Template()
        self.str_buffer = buf if buf is not None else StringIO()
        self.ctx = ctx if ctx is not None else Context(self.str_buffer, **{})

    def render(self, filename=None):
        # TODO: Make this file/filename agnostic
        if (self.str_buffer is not ""):
            self.str_buffer = StringIO()  # Refresh the buffer
        self.template.render_context(self.mako_context)


class MyriadModule(object):

    @staticmethod
    def gen_typedef_decl(my_fun):
        pass

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

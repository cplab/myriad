"""
struct MyriadObject
{
    const struct MyriadClass* m_class; //! Object's class/description
};

struct MyriadClass
{
    const struct MyriadObject _;               //! Embedded object
    const struct MyriadClass* super;           //! Super Class
    const struct MyriadClass* device_class;    //! On-device class
    size_t size;                               //! Object size
    ctor_t my_ctor;                            //! Constructor
    dtor_t my_dtor;                            //! Destructor
    cudafy_t my_cudafy;                        //! CUDAfier
    de_cudafy_t my_decudafy;                   //! De-CUDAficator
};
"""
from mako.runtime import Context
from mako.template import Template
from StringIO import StringIO


class _MakoTemplate(object):

    def __init__(self, template: '', buf=None, ctx=None):
        self.template = template if template is not None else Template()
        self.str_buffer = buf if buf is not None else StringIO()
        self.ctx = ctx if ctx is not None else Context(self.str_buffer, **{})

    def render(self, filename=None):
        # TODO: Make this file/filename agnostic
        if (self.str_buffer is not ""):
            self.str_buffer = StringIO()  # Refresh the buffer
        self.template.render_context(self.mako_context)


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

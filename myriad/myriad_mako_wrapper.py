"""
Common wrapper for Mako templates
"""
import os
from io import StringIO

from mako.template import Template
from mako.runtime import Context
from mako import exceptions

from .myriad_utils import enforce_annotations


class MakoTemplate(object):
    """ Wraps a mako template, context, and I/O Buffer. """

    @enforce_annotations
    def __init__(self, template, context=None, buf: StringIO=None):
        """ Initializes a template relevant data """
        # Sets template
        self._template = None
        if isinstance(template, str):
            self._template = Template(template)
        elif isinstance(template, Template):
            self._template = Template
        else:
            raise TypeError("Invalid template: expected string or Template")

        # Set internal string buffer
        self._buffer = buf
        if buf is None:
            self._buffer = StringIO()

        # Set context and context variables
        context = context if context is not None else {}
        self._context = Context(self._buffer, **context)

    @property
    def buffer(self) -> str:
        """ Returns contents of string buffer """
        return self._buffer.getvalue()

    def reset_buffer(self):
        """ Refreshes the internal buffer and resets the context """
        self._buffer = StringIO()
        self._context = Context(self._buffer, **self._context.kwargs)

    @property
    def context(self) -> dict:
        """ Returns a copy of the internal context namespace as a dict """
        return self._context.kwargs

    @context.setter
    def context(self, new_context: dict):
        """ Replaces current context with new context and refreshes buffer """
        self._buffer = StringIO()
        self._context = Context(self._buffer, **new_context)

    def render(self):
        """ Renders the template to the internal buffer."""
        try:
            self._template.render_context(self._context)
        except exceptions.MakoException:
            print(exceptions.text_error_template().render())


class MakoFileTemplate(MakoTemplate):
    """ A MakoTemplate wrapper with file I/O functionality. """

    @enforce_annotations
    def __init__(self,
                 filename: str,
                 template,
                 context=None,
                 buf: StringIO=None):
        """ Initializes a template relevant data """

        # Sets filename
        self.filename = filename

        # Superclass does the rest of the work for us
        super().__init__(template, context, buf)

    @enforce_annotations
    def render_to_file(self, filename: str=None, overwrite: bool=True):
        """
        Renders the template to a file with the given filename
        """
        if filename is None:
            filename = self.filename
        if not overwrite and os.path.isfile(filename):
            return
        self.render()
        with open(filename, 'w') as filep:
            filep.write(self._buffer.getvalue())

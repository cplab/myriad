#!/usr/bin/python3
"""
Common wrapper for Mako templates
"""

from m_annotations import enforce_annotations
from mako.template import Template
from mako.runtime import Context
from io import StringIO


class MakoTemplate(object):
    """ Wraps a mako template, context, and I/O buffer """

    @enforce_annotations
    def __init__(self,
                 template,
                 context: dict=None,
                 buf: StringIO=None):
        """ Initializes a template with a template, buffer, and context """

        # Sets template
        self._template = None
        if type(template) is str:
            self._template = Template(template)
        elif type(template) is Template:
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
    @enforce_annotations
    def context(self, new_context: dict):
        """ Replaces current context with new context & refreshes buffer."""
        self._buffer = StringIO()
        self._context = Context(self._buffer, **new_context)

    def render(self):
        """ Renders the template """
        self._template.render_context(self._context)


def main():
    """ Renders a simple MakoTemplate """
    tmp = MakoTemplate("hello ${data}!")
    tmp.context = {"data": "World"}
    tmp.render()
    print(tmp.buffer)

if __name__ == "__main__":
    main()

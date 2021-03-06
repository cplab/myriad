"""
High level function assembler. Ties together ast_parse to parse
a full function.

"""
import inspect
import re
from types import FunctionType
from ast import parse
from collections import OrderedDict

from .ast_parse import parse_node
from .myriad_ctypes import CForLoop, CAssign, CVarAttr, CObject, CList
from .myriad_ctypes import get_node_from_var
from .myriad_types import MyriadScalar, MyriadFunction, MVoid, MyriadCType
from .myriad_utils import remove_header_parens, indent_fix_lines

__author__ = ["Alex J. Davies", "Pedro Rittner"]
FLOAT_TYPE = "double"

# Attributes cannot be referenced before their parent object.
# Lists must be assigned explicitly and reassigned explicitly.


class CFunc(object):
    """ Function abstraction object """

    def __init__(self, pyCode):
        self.variables = {}
        self.variableValues = {}
        self.lists = []
        self.nodeList = []
        self.pyCode = pyCode

    def parse_python(self):
        parsed = parse(self.pyCode).body
        for node in parsed:
            convertedCType = parse_node(node)
            self.nodeList.append(convertedCType)

    def stringify_declarations(self):
        retString = ""
        for var in self.variables:
            retString = str(self.variables[var]) + " " + str(var) + ";\n"
        return retString

    # TODO: support for uint, void, sizet, va_list, strings?
    def stringify(self):
        retString = self.stringify_declarations()

        for node in self.nodeList:
            if isinstance(node, CForLoop):
                retString = retString + node.stringify(self.lists) + "\n"
            else:
                retString = retString + node.stringify() + "\n"
        return retString

    def track_variables(self, l):
        # TODO: populate flavours
        flavours = {type(1): "int_fast32_t", type(3.0): FLOAT_TYPE}
        for node in l:
            if (isinstance(node, CAssign) and
                    not isinstance(node.target, CVarAttr) and
                    node.target.var not in self.variables):
                targetVar = node.target.var
                # TODO: what about variable to variable assignments?
                varType = flavours[type(node.val)]
                self.variables.update({targetVar: varType})
            elif isinstance(node, CObject):
                self.track_variables(list(node.__dict__.values()))
            elif isinstance(node, list):
                self.track_variables(node)

    # TODO: write tie_variables(self, l) to tie variables to initial values
    # Easily done because all variables are the first instance of themselves.
    # Just look at their CAssign state ment container.

    def prepare_stringify(self):
        """
        Prepare CFunc for stringification.
        """
        self.track_variables(self.nodeList)
        # self.track_attributes(self.nodeList)
        self.track_lists(self.nodeList)
        self.tie_lists(self.variables, self.lists)

    def track_attributes(self, l):
        for node in l:
            if isinstance(node, CVarAttr):
                target = get_node_from_var(self.variables, node.var)
                if node.attr not in target.attributes:
                    target.attributes.append(node.attr)
            elif isinstance(node, CObject):
                self.track_attributes(list(node.__dict__.values()))

    def track_lists(self, l):
        # TODO: rerun this when updating any lists.
        # Erase self.lists and repopulate.
        for node in l:
            if (isinstance(node, CAssign) and
                    isinstance(node.val, CList)):
                self.lists.append([node.target, node.val])
            elif isinstance(node, CObject):
                self.track_variables(list(node.__dict__.values()))
            elif isinstance(node, list):
                self.track_variables(node)

    def tie_lists(self, variables, lists):
        # Needs to only get first list values so that we have a source for the
        # list declaration.
        for lPair in lists:
            for node in variables:
                if node.var == lPair[0].var:
                    lPair[0] = node


def pyfunbody_to_cbody(c_fun: FunctionType,
                       c_methods=None,
                       struct_members: OrderedDict=None) -> str:
    # Get function source and remove function header
    fun_source = inspect.getsourcelines(c_fun)[0]
    fun_source = remove_header_parens(fun_source)

    # Do some string processing aerobics for indent purposes
    fun_body = indent_fix_lines(fun_source)

    # Parse function body into C string
    fun_parsed = CFunc(fun_body)

    fun_parsed.parse_python()

    fun_parsed.prepare_stringify()

    fun_cstring = fun_parsed.stringify()

    # Add struct pointer cast to self-> instances, if this function is a method
    # TODO: Change this to take an external argument instead
    if re.compile(r".+\.").match(c_fun.__qualname__) is not None:
        repl = "((struct " + c_fun.__qualname__.split('.')[0] + "*) self)->"
        fun_cstring = fun_cstring.replace("self->", repl)

    return fun_cstring


def pyfun_to_cfun(fun: FunctionType, verbatim: bool=False) -> MyriadFunction:
    """
    Converts a native python function into an equivalent C function, 1-to-1.

    Args:
        fun (FunctionType):  function to be converted
        indent_lvl: level of indentation of the function

    Returns:
       MyriadFunction.  converted C function definition
    """
    # Process the function header
    fun_name = fun.__name__

    # Process parameters w/ annotations
    # Need to call copy because parameters is a weak reference proxy
    fun_parameters = inspect.signature(fun).parameters.copy()
    # We can't legally iterate over the original, and items() returns a view
    for argname, argtype in fun_parameters.copy().items():
        # First argument is always self, a void*
        if argname == "self":
            fun_parameters[argname] = MyriadScalar("self", MVoid, ptr=True)
            continue
        # We can do this because the original key insert position is unchanged
        if issubclass(argtype.annotation.__class__, MyriadCType):
            fun_parameters[argname] = MyriadScalar(argname, argtype.annotation)
        elif isinstance(argtype.annotation, MyriadScalar):
            fun_parameters[argname] = argtype.annotation

    # Process return type: if empty, use MVoid
    fun_return_type = inspect.signature(fun).return_annotation
    if fun_return_type is not inspect.Signature.empty:
        if issubclass(fun_return_type.__class__, MyriadCType):
            fun_return_type = MyriadScalar("_", fun_return_type)
    else:
        fun_return_type = MyriadScalar("_", MVoid)

    # Get function body; verbatim methods have C code in their docstrings
    fun_body = fun.__doc__ if verbatim else pyfunbody_to_cbody(fun)

    # Create MyriadFunction wrapper
    myriad_fun = MyriadFunction(fun_name,
                                fun_parameters,
                                fun_return_type,
                                fun_def=fun_body)
    return myriad_fun

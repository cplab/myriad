"""
High level function assembler. Ties together ast_parse to parse
a full function.
Author: Alex Davies
"""

import inspect
from types import FunctionType
from ast import parse

import ast_parse
import CTypes
import myriad_types

floatType = "double"

# Attributes cannot be referenced before their parent object.
# Lists must be assigned explicitly and reassigned explicitly.

class CFunc(object):

    def __init__(self, pyCode):
        self.variables = {}
        self.variableValues = {}
        self.lists = []
        self.nodeList = []
        self.pyCode = pyCode

    def parse_python(self):
        parsed = parse(self.pyCode).body
        for node in parsed:
            convertedCType = ast_parse.parse_node(node)
            self.nodeList.append(convertedCType) 

    def stringify_declarations(self):
        retString = ""
        for var  in self.variables:
            retString = self.variables[var] + " " + var + ";\n"
        return retString

        #TODO: support for uint, void, sizet, va_list, strings?            
    
    def stringify(self):
        retString = self.stringify_declarations()        

        for node in self.nodeList:
            if isinstance(node, CTypes.CForLoop):
                retString = retString + node.stringify(self.lists) + "\n"
            else:
                retString = retString + node.stringify() + "\n"
        return retString


    # TODO: do variables need to be explicitly used without attributes?
    # Possibly, add CVarAttr tracker.
    """def track_variables(self, l):
        for node in l:
            if isinstance(node, CTypes.CVar):
                tempList = []
                for v in self.variables:
                    tempList.append(v.var)
                if node.var not in tempList:
                    self.variables.append(node)
            elif isinstance(node, CTypes.CObject):
                self.track_variables(list(node.__dict__.values()))
            elif isinstance(node, list):
                self.track_variables(node)"""

    def track_variables(self, l):
        #TODO: populate flavours
        flavours = {type(1) : "int64_t", type(3.0) : floatType}        
        for node in l:
            if isinstance(node, CTypes.CAssign) and node.target.var not in self.variables:
                targetVar = node.target.var
                #TODO: what about variable to variable assignments?
                varType = flavours[type(node.val)]
                self.variables.update({targetVar : varType})
            elif isinstance(node, CTypes.CObject):
                self.track_variables(list(node.__dict__.values()))
            elif isinstance(node, list):
                self.track_variables(node)
                

    # TODO: write tie_variables(self, l) which will tie variables to their initial values.
    # Easily done because all variables are the first instance of themselves.
    # Just look at their CAssign state ment container.

    def prepare_stringify(self):
        """
        Prepare CFunc for stringification.
        """
        self.track_variables(self.nodeList)
        self.track_attributes(self.nodeList)
        self.track_lists(self.nodeList)
        self.tie_lists(self.variables, self.lists)

    def track_attributes(self, l):
        for node in l:
            if isinstance(node, CTypes.CVarAttr):
                target = CTypes.get_node_from_var(self.variables, node.var)
                if node.attr not in target.attributes:
                    target.attributes.append(node.attr)
            elif isinstance(node, CTypes.CObject):
                self.track_attributes(list(node.__dict__.values()))

    def track_lists(self, l):
        # TODO: rerun this when updating any lists. Erase self.lists and repopulate.
        for node in l:
            if (isinstance(node, CTypes.CAssign) and
                    isinstance(node.val, CTypes.CList)):
                self.lists.append([node.target, node.val])
            elif isinstance(node, CTypes.CObject):
                self.track_variables(list(node.__dict__.values()))
            elif isinstance(node, list):
                self.track_variables(node)

    def tie_lists(self, variables, lists):
        # Needs to only get first list values so that we have a source for the list declaration.
        for lPair in lists:
            for node in variables:
                if node.var == lPair[0].var:
                    lPair[0] = node


def pyfunbody_to_cbody(fun: FunctionType, methods=None, indent_lvl=2) -> str:
    # Remove function header and leading spaces on each line
    # How many spaces are removed depends on indent level
    fun_source = inspect.getsource(fun)
    fun_body = []

    for line in fun_source[fun_source.index(":\n")+2:].split("\n"):
        fun_body.append(line[(indent_lvl*4):])
    fun_body = "\n".join(fun_body)

    # Parse function body into C string
    fun_parsed = CFunc(fun_body)

    fun_parsed.parse_python()

    fun_parsed.prepare_stringify()

    return fun_parsed.stringify()


def pyfun_to_cfun(fun: FunctionType) -> myriad_types.MyriadFunction:
    """
    Converts a native python function into an equivalent C function, 1-to-1.

    Args:
       fun (FunctionType):  function to be converted

    Returns:
       MyriadFunction.  converted C function definition
    """
    # Process the function header
    fun_name = fun.__name__

    # Process parameters w/ annotations
    # Need to call copy because parameters is a weak reference proxy
    fun_parameters = inspect.signature(fun).parameters.copy()
    # We can't legally iterate over the original, and items() returns a view
    for key in fun_parameters.copy().keys():
        # We can do this because the original key insert position is unchanged
        fun_parameters[key] = fun_parameters[key].annotation
    # print(fun_parameters)

    # Process return type: if empty, use MVoid
    fun_return_type = inspect.signature(fun).return_annotation
    if fun_return_type is inspect.Signature.empty:
        fun_return_type = myriad_types.MyriadScalar("_", myriad_types.MVoid)

    # Get function body
    fun_body = pyfunbody_to_cbody(fun, indent_lvl=1)

    # Create MyriadFunction wrapper
    myriad_fun = myriad_types.MyriadFunction(fun_name,
                                             fun_parameters,
                                             fun_return_type,
                                             fun_def=fun_body)
    return myriad_fun

    # TODO: Problem: Stripping spaces doesn't work for sub functions.



def fun(a: myriad_types.MyriadScalar("a", myriad_types.MInt),
        b: myriad_types.MyriadScalar("b", myriad_types.MInt)):
    x = 0
    while x < 3:
        if a == b:
            x = x + 1
            #c[x] = 1
    return c

#TODO: sort out assignments like the commented line in test


def main():
    mfun = pyfun_to_cfun(fun)
    print(mfun.stringify_decl())
    print("{")
    print(mfun.stringify_def())
    print("}")

if __name__ == "__main__":
    main()

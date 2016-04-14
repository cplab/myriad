"""
Functions for parsing Python string to CTypes using the CFunc framework.
Author: Alex Davies
"""

from .myriad_ctypes import CForLoop, CAssign, CVar, CCompare, CVarAttr, CCall
from .myriad_ctypes import CList, CChar, CString, CSubscript, CUnaryOp
from .myriad_ctypes import CBinaryOp, CReturn, CIf, CWhileLoop, CBoolOp

# TODO: add comparison chaining
# TODO: implement function calls for simple mathematical functions
# TODO: add certain mathematical functions ready made. They put in a dummy call
#   and the call to the C function is stringified.

"""
Pointers are built-in with the Starred node. Yay.
How do we inform users about pointers though?
Build lists of variables to be initialised.
Build lists of functions for typedefs.
Do we need print statements?

Limitations:
* No sets
* No dictionaries
* No string manipulation
* Lists must hold all the same types
* No function calls
* No simple if statements
* No quick incrementing (?)
* No not
* No bit operations
* No is operator
* No augmented assignments
* No breaks
* No use of in
* No consecutive unary ops eg: -+a
* While loops require tracker assignment immediately before loop initaliser
* Single line requirements on many nodes
* Lists must be assigned locally and before use
"""


def parse_node(node):
    literals = ["Num", "Str", "List", "NameConstant"]

    nodeClassName = node.__class__.__name__

    nodeClassDispatch = {"Subscript": parse_subscript,
                         "Name": parse_var,
                         "UnaryOp": parse_unaryop,
                         "BinOp": parse_binaryop,
                         "BoolOp": parse_boolop,
                         "Compare": parse_compare,
                         "Attribute": parse_attribute,
                         "Assign": parse_assign,
                         "For": parse_for_loop,
                         "While": parse_while_loop,
                         "If": parse_if_statement,
                         "Return": parse_return,
                         "Call": parse_call}

    if nodeClassName in literals:
        return parse_literal(node)
    if nodeClassName == "Expr":
        return parse_node(node.value)
    else:
        return nodeClassDispatch[nodeClassName](node)


def parse_literal(node):

    # No sets
    # No dictionaries
    # No string manipulation
    # Lists must hold all the same types

    nodeClassName = node.__class__.__name__

    if nodeClassName == "Num":
        return node.n
    elif nodeClassName == "Str":
        if len(node.s) == 1:
            return CChar(node.s)
        else:
            return CString(node.s)
    elif nodeClassName == "List":
        retList = []
        for n in node.elts:
            retList.append(parse_literal(n))
        return CList(retList)
    elif nodeClassName == "NameConstant":
        return node.value


def parse_subscript(node):
    return CSubscript(node.value, parse_node(node.slice.value))


def parse_var(node):
    """ Parses variable node """
    if node.__class__.__name__ == "Name":
        return CVar(node)


def parse_call(node):
    """ Parses function call """
    args = []
    for arg in node.args:
        args.append(parse_node(arg))
    func = parse_node(node.func)
    return CCall(func, args)


def parse_unaryop(node):
    """ Parses unary operator """
    return CUnaryOp(node, parse_node(node.operand))


def parse_binaryop(node):
    """ Parses binary operator """
    lhs = parse_node(node.left)
    rhs = parse_node(node.right)
    return CBinaryOp(node, lhs, rhs)


def parse_boolop(node):
    """ Parses boolean operator """
    vals = []
    for val in node.values:
        vals.append(parse_node(val))
    return CBoolOp(node, vals)


def parse_compare(node):
    """ Parses comparison operator """
    # nodeOp = node.ops[0].__class__.__name__
    left = parse_node(node.left)
    comparator = parse_node(node.comparators[0])
    return CCompare(node, left, comparator)


def parse_attribute(node):
    """ Parses attribute retrieval """
    return CVarAttr(node)


def parse_assign(node):
    """ Parses assignment operator """
    target = parse_node(node.targets[0])
    val = parse_node(node.value)
    return CAssign(target, val)


def parse_for_loop(node):
    """ Parses for loop declaration """
    # TODO: Ensure that list has been defined locally
    target = parse_node(node.target)
    iterateOver = parse_node(node.iter)
    body = []
    for child in node.body:
        newNode = parse_node(child)
        body.append(newNode)
    return CForLoop(target, iterateOver, body)


def parse_while_loop(node):
    """ Parses while loop declaration """
    # TODO: Ensure that temporary variable is assigned
    cond = parse_node(node.test)
    body = []
    for child in node.body:
        newNode = parse_node(child)
        body.append(newNode)
    return CWhileLoop(cond, body)


def parse_if_statement(node):
    """ Parses if/else statement """
    cond = parse_node(node.test)
    true = []
    for child in node.body:
        newNode = parse_node(child)
        true.append(newNode)
    false = []
    for child in node.orelse:
        newNode = parse_node(child)
        false.append(newNode)
    return CIf(cond, true, false)


def parse_return(node):
    """ Parses return statement """
    return CReturn(parse_node(node.value))


def determine_type(tvar):
    """ Utility function for determining the type of tvar """
    tType = type(tvar)
    if tType is int:
        return "int"
    elif tType is float:
        return "float"
    elif tType is str and len(tvar) == 1:
        return "char"
    elif tType is str and len(tvar) != 1:
        return "str"
    elif tType is list:
        return "list"
    elif tType is tuple:
        return "tuple"

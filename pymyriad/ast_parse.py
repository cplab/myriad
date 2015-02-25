"""
Functions for parsing Python string to CTypes using the CFunc framework.
Author: Alex Davies
"""

import ast
import CTypes

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
            return CTypes.CChar(node.s)
        else:
            return CTypes.CString(node.s)
    elif nodeClassName == "List":
        retList = []
        for n in node.elts:
            retList.append(parse_literal(n))
        return CTypes.CList(retList)   
    elif nodeClassName == "NameConstant":
        return node.value


def parse_subscript(node):
    return CTypes.CSubscript(node.value, parse_node(node.slice.value))


def parse_var(node):
    nodeClassName = node.__class__.__name__

    if nodeClassName == "Name":
        return CTypes.CVar(node)

def parse_call(node):
 
    args = []
    for a in node.args:
        args.append(parse_node(a))

    func = parse_node(node.func)
    return CTypes.CCall(func, args)


def parse_unaryop(node):
    
    return CTypes.CUnaryOp(node, parse_node(node.operand))


def parse_binaryop(node):
    l = parse_node(node.left)
    r = parse_node(node.right)
    return CTypes.CBinaryOp(node, l, r)


def parse_boolop(node):
    vals = []
    for v in node.values:
        vals.append(parse_node(v))
    return CTypes.CBoolOp(node, vals)


def parse_compare(node):
    nodeOp = node.ops[0].__class__.__name__
    left = parse_node(node.left)
    comparator = parse_node(node.comparators[0])
    return CTypes.CCompare(node, left, comparator)


def parse_attribute(node):
    return CTypes.CVarAttr(node)


def parse_assign(node):
    target = parse_node(node.targets[0])
    print(target)
    val = parse_node(node.value)
    return CTypes.CAssign(target, val)


# Ensure that list has been defined locally
def parse_for_loop(node):
    target = parse_node(node.target)
    iterateOver = parse_node(node.iter)
    body = []
    for child in node.body:
        newNode = parse_node(child)
        body.append(newNode)
    return CTypes.CForLoop(target, iterateOver, body)


# Ensure that temporary variable is assigned
def parse_while_loop(node):
    cond = parse_node(node.test)
    body = []
    for child in node.body:
        newNode = parse_node(child)
        body.append(newNode)
    return CTypes.CWhileLoop(cond, body)


def parse_if_statement(node):
    cond = parse_node(node.test)
    true = []
    for child in node.body:
        newNode = parse_node(child)
        true.append(newNode)
    false = []
    for child in node.orelse:
        newNode = parse_node(child)
        false.append(newNode)
    return CTypes.CIf(cond, true, false)


def parse_return(node):
    return CTypes.CReturn(parse_node(node.value))


def determine_type(t):
    tType = type(t)
    if tType is int:
        return "int"
    elif tType is float:
        return "float"
    elif tType is str and len(t) == 1:
        return "char"
    elif tType is str and len(t) != 1:
        return "str"
    elif tType is list:
        return "list"
    elif tType is tuple:
        return "tuple"

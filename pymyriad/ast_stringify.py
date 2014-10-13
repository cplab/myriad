import ast
import CTypes

# TODO: add comparison chaining
# TODO: implement function calls for simple mathematical functions
# TODO: add certain mathematical functions ready made. They put in a dummy call
#   and the call to the C function is stringified.
# TODO: sort line endings

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
* While loops require tracker assignment immediately before loop initaliser
* Single line requirements on many nodes,
"""


def stringify_node(node):
    literals = ["Num", "Str", "List", "NameConstant"]

    nodeClassName = node.__class__.__name__

    nodeClassDispatch = {"Subscript": stringify_subscript,
                         "Expr": stringify_node,
                         "Name": stringify_var,
                         "UnaryOp": stringify_unaryop,
                         "BinOp": stringify_binaryop,
                         "Compare": stringify_compare,
                         "Attribute": stringify_attribute,
                         "Assign": stringify_assign,
                         "For": stringify_for_loop,
                         "While": stringify_while_loop,
                         "If": stringify_if_statement,
                         "Return": stringify_return}

    if nodeClassName in literals:
        return stringify_literal(node)
    else:
        return nodeClassDispatch[nodeClassName](node)


def stringify_literal(node):

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
            retList.append(stringify_literal(n))
        return CTypes.CList(retList)   
    elif nodeClassName == "NameConstant":
        return node.value


def stringify_subscript(node):
    return CTypes.CSubscript(node.value, node.slice)


def stringify_var(node):
    nodeClassName = node.__class__.__name__

    if nodeClassName == "Name":
        return CTypes.CVar(node)


def stringify_unaryop(node):
    
    return CTypes.CUnaryOp(node, stringify_node(node.operand))


def stringify_binaryop(node):
    l = stringify_node(node.left)
    r = stringify_node(node.right)
    return CTypes.CBinaryOp(node, l, r)


def stringify_boolop(node):
    vals = []
    for v in node.values:
        vals.append(stringify_node(v))
    return CTypes.CBoolOp(node, vals)


def stringify_compare(node):
    nodeOp = node.ops[0].__class__.__name__
    left = stringify_node(node.left)
    comparator = stringify_node(node.comparators[0])
    return CTypes.CCompare(node, left, comparator)


def stringify_attribute(node):
    return CTypes.CVarAttr(node)


def stringify_assign(node):
    target = stringify_node(node.targets[0])
    val = stringify_node(node.value)
    return CTypes.CAssign(target, val)


def stringify_for_loop(node):
    target = stringify_node(node.target)
    iterateOver = stringify_node(node.iter)
    body = []
    for child in node.body:
        newNode = stringify_node(child)
        body.append(newNode)
    return CTypes.CForLoop(target, iterateOver, body)


def stringify_while_loop(node):
    cond = stringify_node(node.test)
    body = []
    for child in node.body:
        newNode = stringify_node(child)
        body.append(newNode)
    return CTypes.CWhileLoop(cond, body)


def stringify_if_statement(node):
    cond = stringify_node(node.test)
    true = []
    for child in node.body:
        newNode = stringify_node(child)
        true.append(newNode)
    false = []
    for child in node.orelse:
        newNode = stringify_node(child)
        false.append(newNode)
    return CTypes.CIf(cond, true, false)


def stringify_return(node):
    return CTypes.CReturn(stringify_node(node.value))


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

# TODO: This is broken; there is no num_stringify anywhere
def test():
    node = ast.parse("1", mode="exec")
    num_stringify(node)

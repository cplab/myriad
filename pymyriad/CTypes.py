"""
Module providing classes for holding parsed Python information.
Author: Alex Davies
"""


class CObject(object):
    """
    Parent object for typechecking purposes.
    """
    def __init__(self):
        pass

    def stringify(self):
        """
        Return the C interpretation of this object as a string.
        """
        raise NotImplementedError("Stringify is an abstract method.")


class CList(CObject):
    """
    List container class.

    Constraints:
    - All elements must be of the same C type
    - Lists must be of fixed length
    """
    
    def __init__(self, l: list):
        """
        Initializes the CList with the given list.

        :param l: list which we are wrapping
                """
        if not isinstance(l, list):
            raise TypeError("l must be set to a list.")
        self.cargo = l  # List itself
        self.cargoType = determine_type(l[0])  # Element type (eg char)
        self.length = len(l)  # Static length of list

        # TODO: Consider whether we need to keep track of this.
        self.numStringifyCalls = 0

    def stringify(self):
        """
        Returns the literal C representation of this list.

        Example:
        [1, 2, 3] >>> {1, 2, 3}
        """
        retString = "{" + stringify(self.cargo[0])
        for elt in self.cargo[1:]:
            retString = retString + ", " + stringify(elt)
        retString = retString + "}"
        return retString


class CSubscript(CObject):
    """
    CSubscript acts as a container for, nominally, indexing by value (e.g.
    container[index]) or indexing by range (e.g. container[i:j]). However,
    slicing is not valid C code; this will be removed in a future version.
    """

    def __init__(self, variableNode, sliceNode):
        """
        Initializes a CSubscript container.

        :param variableNode: variable on which the subscript is acting on
        :param sliceNode: indicates whether it's an index or a slice
        """
        self.val = variableNode.id  # String identified of the variable
        self.sliceClass = sliceNode.__class__.__name__  # Name of slice class
        self.sl = sliceNode
        """
        # TODO: fix for variables (e.g. l[a] where a is a variable)
        # The above is necessary for loops.
        if self.sliceClass == "Index":
            #indexValue = sliceNode.value.n
            self.sliceDict = {"Index" : indexValue}

        if self.sliceClass == "Slice":
            lowerValue = sliceNode.lower.n
            upperValue = sliceNode.upper.n
            self.sliceDict = {"Lower" : lowerValue, "Upper" : upperValue}
        """

    def stringify(self):
        """
        In this case, the stringification results in things like l[0].
        """
        return str(self.val) + "[" + stringify(self.sl) + "]"


class CChar(CObject):
    """
    CChars are strings of one element, where their C representations are
    single-quoted.
    """

    def __init__(self, c: str):
        if ((not isinstance(c, str)) or (len(c) != 1)):
            raise TypeError("s must be set to a string of length = 1.")

        self.cargo = c
        self.length = 1

    def stringify(self):
        return str("'" + self.cargo + "'")


class CString(CObject):
    """
    CString are string literals identical to CChars but using double quotes
    and having fixed lengths greater than 1.
    """

    def __init__(self, s: str):
        if not isinstance(s, str):
            raise TypeError("s must be set to a string.")

        self.cargo = s
        self.length = len(s)

    def stringify(self):
        return str('"' + self.cargo + '"')


class CVar(CObject):
    """
    Container that represents a variable and its identifier. Additionally,
    it contains the context in which it exists.
    """

    def __init__(self, variableNode):
        self.var = variableNode.id
        self.ctx = variableNode.ctx.__class__.__name__
        # Attributes are the members of the variable.
        self.attributes = []

    def stringify(self):
        return str(self.var)


class CVarAttr(CVar):
    """
    Container for an attribute for a variable (e.g. a.foo).
    """
    def __init__(self, node):
        if node.__class__.__name__ == "Name":
            self.var = CVar(node.value)
            self.attr = node.attr
        elif node.__class__.__name__ == "Attribute":
            if node.value.__class__.__name__ == "Name":
                self.var = CVar(node.value)
                self.attr = node.attr
            elif node.value.__class__.__name__ == "Attribute":
                self.var = CVarAttr(node.value)
                self.attr = node.attr

    def stringify(self):
        return str(self.var.var) + "->" + str(self.attr)


class CCall(CObject):
    """
    Container for a function call.
    """

    funcDict = {"acos": "acos", "asin": "asin", "atan": "atan",
                "atan2": "atan2", "ceil": "ceil", "cos": "cos",
                "cosh": "cosh", "exp": "exp", "fabs": "fabs",
                "floor": "floor", "fmod": "fmod", "frexp": "frexp",
                "ldexp": "ldexp", "log": "log", "log10": "log10",
                "modf": "modf", "sin": "sin", "sinh": "sinh",
                "sqrt": "sqrt", "tan": "tan", "tanh": "tanh",
                "erf": "erf", "erfc": "erfc", "lgamma": "lgamma",
                "hypot": "hypot", "isnan": "isnan", "acosh": "acosh",
                "asinh": "asinh", "atanh": "atanh", "expm1": "expm1",
                "log1p": "log1p"}

    # No Bessel functions (j0, j1, jn, y0, y1, yn).
    # No cube root (cbrt).
    # No ilogb, logb.
    # No nextafter.
    # No remainder.
    # No rint.
    # No scalb.

    def __init__(self, func, args):
        # If our alleged function call is ...
        # ... a standalone function call
        if func.var in self.funcDict:
            self.func = self.funcDict[func.var]
            self.args = args
        # ... a call to a self method
        elif isinstance(func.var, CVar) and func.var.var == 'self':
            self.func = func.attr
            self.args = [func.var] + args
        # ... something else
        else:
            raise Exception("Function call not valid.")

    def stringify(self):
        retString = stringify(self.func)
        if not self.args:
            return retString + "() "
        else:
            retString = retString + "(" + stringify(self.args[0])
            i = 1
            while i < len(self.args):
                retString = retString + ", " + stringify(self.args[i])
                i += 1
                return retString + ");"


class CUnaryOp(CObject):
    """
    Container for simple unary operations (add, subtract, not).
    """

    def __init__(self, unaryOpNode, operand):
        self.operand = operand
        nodeOp = unaryOpNode.op.__class__.__name__

        if nodeOp == "UAdd":
            self.op = "+"
        if nodeOp == "USub":
            self.op = "-"
        if nodeOp == "Not":
            self.op = "!"

    def stringify(self):
        return self.op + self.operand.stringify()


class CBinaryOp(CObject):
    """
    Container for simple binary operations (add, subtract, multiply, mod).
    """
    def __init__(self, node, left, right):
        """
        :param nodeOp: actual operator stringified for C
        :param left: LHS of the operation, either a CVar or a literal
        :param right: RHS of the operation, either a CVar or a literal
        """
        self.left = left
        self.right = right
        nodeOp = node.op.__class__.__name__

        if nodeOp == "Add":
            self.op = "+"
        if nodeOp == "Sub":
            self.op = "-"
        if nodeOp == "Mult":
            self.op = "*"
        if nodeOp == "Div":
            self.op = "/"
        if nodeOp == "Mod":
            self.op = "%"
        if nodeOp == "Pow":
            self.op = "**"

    def stringify(self):
        """
        Returns the literal representation of the operator, except in
        the case of the power operator which is replace with a function
        call to <math.h>'s pow() function.
        """
        if self.op == "**":
            return str("pow(" + stringify(self.left) + ", " + stringify(self.right) + ")")
        return str(stringify(self.left) + " " + self.op + " " + stringify(self.right))


class CBoolOp(CObject):
    """
    Container for simple bit-wise boolean operations.
    """
    def __init__(self, node, vals):
        nodeOp = node.op.__class__.__name__

        if nodeOp == "Or":
            self.op = "||"
        if nodeOp == "And":
            self.op = "&&"
        self.vals = vals

    def stringify(self):
        retString = stringify(self.vals[0])
        for node in self.vals[1:]:
            retString = retString + " " + self.op + " " + stringify(node)
        return retString 


class CCompare(CObject):
    """
    Container for comparison operations (greater than, etc.).
    """
    def __init__(self, node, l, r):
        nodeOp = node.ops[0].__class__.__name__

        if nodeOp == "Eq":
            self.op = "=="
        if nodeOp == "NotEq":
            self.op = "!="
        if nodeOp == "Lt":
            self.op = "<"
        if nodeOp == "LtE":
            self.op = "<="
        if nodeOp == "Gt":
            self.op = ">"
        if nodeOp == "GtE":
            self.op = ">="
        # TODO: What is the C implementation for this? There isn't one.
        # Just use a for loop iterating over the list, linear search.
        if nodeOp == "In":
            self.op = "in"
        self.left = l
        self.right = r

    def stringify(self):
        return str(stringify(self.left) + " " + self.op + " " + stringify(self.right))


class CAssign(CObject):
    """
    Container for any arbitrary RHS => LHS assignment, where LHS is a var.
    RHS may be a variable, literal, expression, or member.
    """
    def __init__(self, t, val):
        self.target = t
        self.val = val

    def stringify(self):
        return str(stringify(self.target) + " = " + stringify(self.val)) + ";"
        # Assignment is always single line.


# TODO: not allow for loops - while loops only
# TODO: implement length functions

class CForLoop(CObject):
    """
    Container for a for loop, using an integer to loop through the data
    structure provided.
    """

    def __init__(self, t, i, b):
        """
        :param target: temporary variable for iteration purposes (eg i)
        :param i: identifier of the list we're looping over
        :param body: list of statements within the for loop
        """
        self.target = t
        self.iterateOver = i
        self.body = b

    def stringify(self, lists):
        """
        Must be called with lists list supplied.

        The C implementation of the for loop requires knowing the
        length of the list, therefore we must find the length of the
        list we're looping over by searching for the relevant
        [variable, list] inside of the master list (i.e. lists).

        :param lists: master list of lists, of elements [var, list]
        """
        iterateOverLPair = get_lPair_from_var(lists, self.iterateOver.var)
        length = iterateOverLPair[1].length
        initialString = "for (int64_t i = 0; " + "i < " + str(length) + "; i++;)"
        bodyString = "{"
        for node in self.body:
            bodyString = bodyString + "\n" + stringify(node)
        bodyString = bodyString + "\n" + "}"
        return initialString + "\n" + bodyString + "\n"


class CWhileLoop(CObject):
    """
    Container for a while loop declaration.
    """
    def __init__(self, c: CCompare, b: list):
        self.cond = c  # condition we're looping over
        self.body = b  # list of C type statements

    def set_tracker(self, var):
        """
        TODO: Check if we need this.
        """
        self.tracker = var

    def stringify(self):
        initialString = "while (" + self.cond.stringify() + ")"
        bodyString = "{"
        for node in self.body:
            bodyString = bodyString + "\n" + stringify(node)
        bodyString = bodyString + "\n" + "}"
        return initialString + "\n" + bodyString + "\n"


class CIf(CObject):
    """
    Container for if statements.

    elif are if statements within the false branch list.
    """

    def __init__(self, c, b, f):
        self.cond = c  # Condition for if
        self.true = b  # Path to go if True
        self.false = f  # Path to go if false

    def stringify(self):
        initialString = "if (" + stringify(self.cond) + ")\n"
        trueString = initialString + "{\n"
        for node in self.true:
                trueString = trueString + stringify(node) + "\n"

        if len(self.false) > 0:
                falseString = "}else{\n"
                for node in self.false:
                        falseString = falseString + stringify(node) + "\n}"
        else:
                falseString = "}"
        return trueString + falseString


class CReturn(CObject):

    def __init__(self, v):
        self.val = v

    def stringify(self):
        return "return " + stringify(self.val) + ";"


def stringify(node) -> str:
    """ Returns either a literal or a stringified C type string."""
    if not isinstance(node, CObject):
            return str(node)
    return node.stringify()


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


def get_node_from_var(l, var):
    """
    Finds the master variable which has identifier var, and returns the
    CVar representing it.

    :param l: master list of variables
    :param var: identifier we are looking for
    """
    for node in l:
        if isinstance(node, CVar) and (node.var == var):
            return node
        elif isinstance(node, list):
            return get_node_from_var(node, var)
    return None


def get_lPair_from_var(l, var):
    """
    Returns the [variable, list] pair for a given identifier var.
    """
    for node in l:
        if isinstance(node, list) and (node[0].var == var):
            return node

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
        self.cargo_type = determine_type(l[0])  # Element type (eg char)
        self.length = len(l)  # Static length of list

    def stringify(self):
        """
        Returns the literal C representation of this list.

        Example:
        [1, 2, 3] >>> {1, 2, 3}
        """
        ret_string = "{" + stringify(self.cargo[0])
        for elt in self.cargo[1:]:
            ret_string = ret_string + ", " + stringify(elt)
        ret_string = ret_string + "}"
        return ret_string


class CSubscript(CObject):
    """
    CSubscript acts as a container for, nominally, indexing by value (e.g.
    container[index]) or indexing by range (e.g. container[i:j]). However,
    slicing is not valid C code; this will be removed in a future version.
    """

    def __init__(self, variable_node, slice_node):
        """
        Initializes a CSubscript container.

        :param variableNode: variable on which the subscript is acting on
        :param sliceNode: indicates whether it's an index or a slice
        """
        super().__init__()
        self.val = variable_node.id  # String identified of the variable
        self.sliceClass = slice_node.__class__.__name__  # Name of slice class
        self.sl = slice_node
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
        if (not isinstance(c, str)) or (len(c) != 1):
            raise TypeError("s must be set to a string of length = 1.")
        super().__init__()
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
        super().__init__()
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
        super().__init__()
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

    C_FUNC_DICT = {
        "acos": "acos",
        "asin": "asin",
        "atan": "atan",
        "atan2": "atan2",
        "ceil": "ceil",
        "cos": "cos",
        "cosh": "cosh",
        "exp": "exp",
        "fabs": "fabs",
        "floor": "floor",
        "fmod": "fmod",
        "frexp": "frexp",
        "ldexp": "ldexp",
        "log": "log",
        "log10": "log10",
        "modf": "modf",
        "sin": "sin",
        "sinh": "sinh",
        "sqrt": "sqrt",
        "tan": "tan",
        "tanh": "tanh",
        "erf": "erf",
        "erfc": "erfc",
        "lgamma": "lgamma",
        "hypot": "hypot",
        "isnan": "isnan",
        "acosh": "acosh",
        "asinh": "asinh",
        "atanh": "atanh",
        "expm1": "expm1",
        "log1p": "log1p"
    }

    # No Bessel functions (j0, j1, jn, y0, y1, yn).
    # No cube root (cbrt).
    # No ilogb, logb.
    # No nextafter.
    # No remainder.
    # No rint.
    # No scalb.

    def __init__(self, func, args):
        super().__init__()
        # If our alleged function call is ...
        # ... a standalone function call
        if func.var in self.C_FUNC_DICT:
            self.func = self.C_FUNC_DICT[func.var]
            self.args = args
        # ... a call to a self method
        elif isinstance(func.var, CVar) and func.var.var == 'self':
            self.func = func.attr
            self.args = [func.var] + args
        # ... something else
        else:
            raise Exception("Function call not valid.")

    def stringify(self):
        ret = stringify(self.func)
        if not self.args:
            return ret + "() "
        else:
            ret = ret + "(" + stringify(self.args[0])
            i = 1
            while i < len(self.args):
                ret = ret + ", " + stringify(self.args[i])
                i += 1
                return ret + ");"


class CUnaryOp(CObject):
    """
    Container for simple unary operations (add, subtract, not).
    """

    def __init__(self, unaryOpNode, operand):
        self.operand = operand
        nodeOp = unaryOpNode.op.__class__.__name__
        self.op = None
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
        self.op = None
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
        super().__init__()
        node_op = node.op.__class__.__name__
        if node_op == "Or":
            self.op = "||"
        if node_op == "And":
            self.op = "&&"
        self.vals = vals

    def stringify(self):
        ret = stringify(self.vals[0])
        for node in self.vals[1:]:
            ret = ret + " " + self.op + " " + stringify(node)
        return ret


class CCompare(CObject):
    """
    Container for comparison operations (greater than, etc.).
    """
    def __init__(self, node, l, r):
        super().__init__()
        node_op = node.ops[0].__class__.__name__
        if node_op == "Eq":
            self.op = "=="
        if node_op == "NotEq":
            self.op = "!="
        if node_op == "Lt":
            self.op = "<"
        if node_op == "LtE":
            self.op = "<="
        if node_op == "Gt":
            self.op = ">"
        if node_op == "GtE":
            self.op = ">="
        # TODO: What is the C implementation for this? There isn't one.
        # Just use a for loop iterating over the list, linear search.
        if node_op == "In":
            self.op = "in"
        self.left = l
        self.right = r

    def stringify(self):
        return str(stringify(self.left) + " " +
                   self.op + " " + stringify(self.right))


class CAssign(CObject):
    """
    Container for any arbitrary RHS => LHS assignment, where LHS is a var.
    RHS may be a variable, literal, expression, or member.
    """
    def __init__(self, t, val):
        super().__init__()
        self.target = t
        self.val = val

    def stringify(self):
        # Assignment is always single line.
        return str(stringify(self.target) + " = " + stringify(self.val)) + ";"


# TODO: implement length functions

class CForLoop(CObject):
    """
    Container for a for loop, using an integer to loop through the data
    structure provided.
    """

    def __init__(self, target, i, body):
        """
        :param target: temporary variable for iteration purposes (eg i)
        :param i: identifier of the list we're looping over
        :param body: list of statements within the for loop
        """
        self.target = target
        self.iterate_over = i
        self.body = body

    def stringify(self, lists):
        """
        Must be called with lists list supplied.

        The C implementation of the for loop requires knowing the
        length of the list, therefore we must find the length of the
        list we're looping over by searching for the relevant
        [variable, list] inside of the master list (i.e. lists).

        :param lists: master list of lists, of elements [var, list]
        """
        llen = get_lpair_from_var(lists, self.iterate_over.var)[1].length
        init_str = "for (int_fast32_t i = 0; " + "i < " + str(llen) + "; i++)"
        body_str = "{"
        for node in self.body:
            body_str = body_str + "\n" + stringify(node)
        body_str = body_str + "\n" + "}"
        return init_str + "\n" + body_str + "\n"


class CWhileLoop(CObject):
    """
    Container for a while loop declaration.
    """
    def __init__(self, c: CCompare, b: list):
        #: condition we're looping over
        self.cond = c
        #: list of C type statements
        self.body = b
        #: #TODO Check if we need this.
        self.tracker = None

    def set_tracker(self, var):
        self.tracker = var

    def stringify(self):
        init_str = "while (" + self.cond.stringify() + ")"
        body_str = "{"
        for node in self.body:
            body_str = body_str + "\n" + stringify(node)
        body_str = body_str + "\n" + "}"
        return init_str + "\n" + body_str + "\n"


class CIf(CObject):
    """
    Container for if statements.

    elif are if statements within the false branch list.
    """

    def __init__(self, c, b, f):
        super().__init__()
        self.cond = c  # Condition for if
        self.true = b  # Path to go if True
        self.false = f  # Path to go if false

    def stringify(self):
        initial_str = "if (" + stringify(self.cond) + ")\n"
        true_str = initial_str + "{\n"
        for node in self.true:
            true_str = true_str + stringify(node) + "\n"
        # Check if we have if/else
        if len(self.false) > 0:
            false_str = "} else {\n"
            for node in self.false:
                false_str = false_str + stringify(node) + "\n}"
        else:
            false_str = "}"
        return true_str + false_str


class CReturn(CObject):
    """ Return statement CObject """
    def __init__(self, v):
        super().__init__()
        self.val = v

    def stringify(self):
        return "return " + stringify(self.val) + ";"


def stringify(node) -> str:
    """ Returns either a literal or a stringified C type string."""
    if not isinstance(node, CObject):
        return str(node)
    return node.stringify()


def determine_type(t_var):
    """ Stringifies the type of a given variable t_var """
    t_type = type(t_var)
    if t_type is int:
        return "int"
    elif t_type is float:
        return "float"
    elif t_type is str and len(t_var) == 1:
        return "char"
    elif t_type is str and len(t_var) != 1:
        return "str"
    elif t_type is list:
        return "list"
    elif t_type is tuple:
        return "tuple"


def get_node_from_var(mlist, var):
    """
    Finds the master variable which has identifier var, and returns the
    CVar representing it.

    :param mlist: master list of variables
    :param var: identifier we are looking for
    """
    for node in mlist:
        if isinstance(node, CVar) and (node.var == var):
            return node
        elif isinstance(node, list):
            return get_node_from_var(node, var)
    return None


def get_lpair_from_var(mlist, var):
    """
    Returns the [variable, list] pair for a given identifier var.
    """
    for node in mlist:
        if isinstance(node, list) and (node[0].var == var):
            return node

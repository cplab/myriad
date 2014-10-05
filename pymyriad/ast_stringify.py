from ast import *
from CTypes import *

#TODO: regex on all strings
	#Answer: ", blah blah, "
#TODO: what if people need external modules?
#TODO: add comparison chaining
#TODO: sort CLine master variables list
#TODO: implement slicing/subscripting
#TODO: use a switch system to determine which stringify method to use
#TODO: write all control flow functions
#TODO: implement function calls for simple mathematical functions
#TODO: work out implementation of while loops - requires identification of index





# Pointers are built-in with the Starred node. Yay.
# How do we inform users about pointers though?
# Convert to MyriadType or just stringify?
# Control flow will be handled with a C style syntax.
# Build lists of variables to be initialised.
# Build lists of functions for typedefs.
# Do we need print statements?
# No need for imports (?)


# No sets
# No dictionaries
# No string manipulation
# Lists must hold all the same types
# No function calls
# No simple if statements
# No quick incrementing (?)
# No not
# No bit operations
# No is operator	
# No augmented assignments
# No breaks
# No use of in	

def stringify_node(node):
	nodeClassName = node.__class__.__name__


	literals = ["Num", "Str", "List", "NameConstant"]
	expressionValues = ["Expr", "Name", "UnaryOp", "BinOp", "BoolOp", "Compare", "Attribute", ]

	if nodeClassName == "Expr":
		return stringify_expr(node)
	if nodeClassName in literals:
		return stringify_literal(node)
	if nodeClassName in expressionValues:
		return stringify_expr_contents(node)
	if nodeClassName == "Assign":
		return stringify_assign(node)
	if nodeClassName == "For":
		return stringify_for_loop(node)
	if nodeClassName == "While":
		return stringify_while_loop(node)
	if nodeClassName == "If":
		return stringify_if_statement(node)
	


def stringify_expr_contents(node):

	literals = ["Num", "Str", "List", "NameConstant"]

	nodeClassName = node.__class__.__name__

	if nodeClassName in literals:
		return stringify_literal(node)
	elif nodeClassName == "Expr":
		return stringify_expr_contents(node.value)
	elif nodeClassName == "Name":
		return stringify_var(node)
	elif nodeClassName == "UnaryOp":
		return stringify_unaryop(node)
	elif nodeClassName == "BinOp":
		return stringify_binaryop(node)
	elif nodeClassName == "BoolOp":
		return stringify_boolop(node)
	elif nodeClassName == "Compare":
		return stringify_compare(node)
	elif nodeClassName == "Attribute":
		return stringify_attribute(node)
	elif nodeClassName == "Assign":
		return stringify_assign(node)
	if nodeClassName == "For":
		return stringify_for_loop(node)
	if nodeClassName == "While":
		return stringify_while_loop(node)
	if nodeClassName == "If":
		return stringify_if_statement(node)
	if nodeClassName == "Return":
		return stringify_return(node)

		
		
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
			return CChar(node.s)
		else:
			return CString(node.s)
	elif nodeClassName == "List":
		retList = []
		for n in node.elts:
			retList.append(stringify_literal(n))
		valueType = determine_type(retList[0])
		return CList(retList)	
	elif nodeClassName == "NameConstant":
		return node.value

def stringify_var(node):
	nodeClassName = node.__class__.__name__

	if nodeClassName == "Name":
		ctx = node.ctx.__class__.__name__
		return CVar(node.id, node.ctx.__class__.__name__)


def stringify_unaryop(node):
	nodeOp = node.op.__class__.__name__
	
	if nodeOp == "UAdd":
		return CUnaryOp("+", stringify_expr_contents(node.operand))
	if nodeOp == "USub":
		return CUnaryOp("-", stringify_expr_contents(node.operand))
	if nodeOp == "Not":
		return CUnaryOp("!", stringify_expr_contents(node.operand))

def stringify_binaryop(node):
	nodeOp = node.op.__class__.__name__
	l = stringify_expr_contents(node.left)
	r = stringify_expr_contents(node.right)

	if nodeOp == "Add":
		return CBinaryOp("+", l, r)
	if nodeOp == "Sub":
		return CBinaryOp("-", l, r)
	if nodeOp == "Mult":
		return CBinaryOp("*", l, r)
	if nodeOp == "Div":
		return CBinaryOp("/", l, r)
	if nodeOp == "Mod":
		return CBinaryOp("%", l, r)
	if nodeOp == "Pow":
		return CBinaryOp("**", l, r)

def stringify_boolop(node):
	nodeOp = node.op.__class__.__name__
	vals = []

	for v in node.values:
		vals.append(stringify_expr_contents(v))

	return CBoolOp(nodeOp, vals)

def stringify_compare(node):
	nodeOp = node.ops[0].__class__.__name__
	left = stringify_expr_contents(node.left)
	comparator = stringify_expr_contents(node.comparators[0])
	
	
	return CCompare(nodeOp, left, comparator)

def stringify_attribute(node):
	
	var = node.value.id
	ctx = node.value.ctx.__class__.__name__
	attr = node.attr
	
	return CVarAttr(var, ctx, attr)

def stringify_assign(node):
	target = stringify_var(node.targets[0])
	val = stringify_node(node.value)

	return CAssign(target, val)
	

	
		


def stringify_for_loop(node):
	
	target = stringify_expr_contents(node.target)
	iterateOver = stringify_expr_contents(node.iter)
	body = []
	for child in node.body:
		newNode = stringify_expr_contents(child)
		body.append(newNode)
	return CForLoop(target, iterateOver, body)

def stringify_while_loop(node):
	cond = stringify_expr_contents(node.test)
	body = []
	for child in node.body:
		newNode = stringify_expr_contents(child)
		body.append(newNode)
	return CWhileLoop(cond, body)

def stringify_if_statement(node):
	cond = stringify_expr_contents(node.test)
	true = []
	for child in node.body:
		newNode = stringify_expr_contents(child)
		true.append(newNode)
	false = []
	for child in node.orelse:
		newNode = stringify_expr_contents(child)
		false.append(newNode)
	return CIf(cond, true, false)

def stringify_return(node):

	return CReturn(stringify_expr_contents(node.value))
	
	
def determine_type(t):
	tType = type(t)
	if tType is int:
		return "int"
	if tType is float:
		return "float"
	if tType is str and length(t) == 1:
		return "char"
	elif tType is str and length(t) != 1:
		return "str"
	if tType is list:
		return "list"
	if tType is tuple:
		return "tuple"

# Pointers are built-in with the Starred node. Yay.
# How do we inform users about pointers though?

# Convert to MyriadType or just stringify?

# Control flow will be handled with a C style syntax.

# Build lists of variables to be initialised.

# Build lists of functions for typedefs.

def test():
	node = parse("1", mode="exec")
	num_stringify(node)




#TODO: use a switch system to determine which stringify method to use


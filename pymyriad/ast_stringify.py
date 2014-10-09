from ast import *
from CTypes import *

#TODO: add comparison chaining
#TODO: implement function calls for simple mathematical functions
#TODO: add certain mathematical functions ready made. They put in a dummy call
#	and the call to the C function is stringified.
#TODO: sort line endings
#TODO: change tabs to spaces.


# Pointers are built-in with the Starred node. Yay.
# How do we inform users about pointers though?
# Build lists of variables to be initialised.
# Build lists of functions for typedefs.
# Do we need print statements?


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
# While loops require tracker assignment immediately before loop initaliser	
# Single line requirements on many nodes,

def stringify_node(node):


	literals = ["Num", "Str", "List", "NameConstant"]

	nodeClassName = node.__class__.__name__

	if nodeClassName in literals:
		return stringify_literal(node)
	elif nodeClassName == "Subscript":
		return stringify_subscript(node)
	elif nodeClassName == "Expr":
		return stringify_node(node.value)
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
	elif nodeClassName == "For":
		return stringify_for_loop(node)
	elif nodeClassName == "While":
		return stringify_while_loop(node)
	elif nodeClassName == "If":
		return stringify_if_statement(node)
	elif nodeClassName == "Return":
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
		return CList(retList)	
	elif nodeClassName == "NameConstant":
		return node.value


def stringify_subscript(node):
	
	return CSubscript(node.value, node.slice)	




def stringify_var(node):
	nodeClassName = node.__class__.__name__

	if nodeClassName == "Name":
		return CVar(node)


def stringify_unaryop(node):
	
	return CUnaryOp(node, stringify_node(node.operand))


def stringify_binaryop(node):
	l = stringify_node(node.left)
	r = stringify_node(node.right)
	
	return CBinaryOp(node, l, r)


def stringify_boolop(node):

	vals = []
	for v in node.values:
		vals.append(stringify_node(v))

	return CBoolOp(node, vals)


def stringify_compare(node):
	nodeOp = node.ops[0].__class__.__name__
	left = stringify_node(node.left)
	comparator = stringify_node(node.comparators[0])
	
	
	return CCompare(node, left, comparator)


def stringify_attribute(node):
	
	
	return CVarAttr(node)


def stringify_assign(node):
	target = stringify_node(node.targets[0])
	val = stringify_node(node.value)

	return CAssign(target, val)
	

def stringify_for_loop(node):
	
	target = stringify_node(node.target)
	iterateOver = stringify_node(node.iter)
	body = []
	for child in node.body:
		newNode = stringify_node(child)
		body.append(newNode)
	return CForLoop(target, iterateOver, body)

def stringify_while_loop(node):
	cond = stringify_node(node.test)
	body = []
	for child in node.body:
		newNode = stringify_node(child)
		body.append(newNode)
	return CWhileLoop(cond, body)

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
	return CIf(cond, true, false)

def stringify_return(node):

	return CReturn(stringify_node(node.value))
	
	
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

def test():
	node = parse("1", mode="exec")
	num_stringify(node)






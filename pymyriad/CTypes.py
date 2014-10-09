

class CObject(object):

	def __init__(self):
		pass



class CList(CObject):
	
	def __init__(self, l):
		if not isinstance(l, list):
			raise TypeError("l must be set to a list.")
		

		self.cargo = l
		self.cargoType = determine_type(l[0])
		self.length = len(l)
		self.numStringifyCalls = 0

	def stringify(self):
		return self.stringify_assignment()

	def stringify_assignment(self):
		retString = "{" + stringify(self.cargo[0])
		for elt in self.cargo[1:]:
			retString = retString + ", " + stringify(elt) 
		retString = retString + "}"
		return retString
	
class CSubscript(CObject):

	def __init__(self, variableNode, sliceNode):
		self.val = variableNode.id
		self.sliceClass = sliceNode.__class__.__name__
		
		#TODO: fix for variables	
	
		if self.sliceClass == "Index":
			indexValue = sliceNode.value.n
			self.sliceDict = {"Index" : indexValue}

		if self.sliceClass == "Slice":
			lowerValue = sliceNode.lower.n		
			upperValue = sliceNode.upper.n
			self.sliceDict = {"Lower" : lowerValue, "Upper" : upperValue}

	def stringify(self):
		
		if self.sliceClass == "Index":
			return str(self.val) + "[" + str(self.sliceDict["Index"]) + "]"
		
		if self.sliceClass == "Slice":
			#TODO: slicing is not valid C.
			return str(self.val) + "[" + str(self.sliceDict["Lower"]) + ", " + str(self.sliceDict["Upper"]) + "]"


class CChar(CObject):
	
	def __init__(self, c):
		if ((not isinstance(c, str)) or (len(c) != 1)):
			raise TypeError("s must be set to a string of length = 1.")

		self.cargo = c
		self.length = 1

	def stringify(self):
		return str("'" + self.cargo + "'")
		

class CString(CObject):
	
	def __init__(self, s):
		if not isinstance(s, str):
			raise TypeError("s must be set to a string.")

		self.cargo = s
		self.length = len(s)

	def stringify(self):
		return str('"' + self.cargo + '"')

class CVar(CObject):
	
	def __init__(self, variableNode):
		self.var = variableNode.id
		self.ctx = variableNode.ctx.__class__.__name__
		self.attributes = []

	def stringify(self):
		return str(self.var)

class CVarAttr(CVar):
	
	def __init__(self, node):
		if node.__class__.__name__ == "Name":
			self.var = CVar(node.value)
			self.attr = node.attr
		elif node.__class__.__name__ == "Attribute":
			if node.value.__class__.__name__ == "Name":
				self.var = CVar(node.value)
			elif node.value.__class__.__name__ == "Attribute":
				self.var = CVarAttr(node.value)
			self.attr = node.attr
			
		
	
	

	#TODO: work out how to implement stringify


class CUnaryOp(CObject):

	
	def __init__(self, unaryOpNode, operand):
		self.operand = operand
		nodeOp = unaryOpNode.__class__.__name__

		if nodeOp == "UAdd":
			self.op = "+"
		if nodeOp == "USub":
			self.op = "-"
		if nodeOp == "Not":
			self.op = "!"
	
	def stringify(self):
		return str(self.op + stringify(self.operand))
	
	
class CBinaryOp(CObject):
	
	def __init__(self, nodeOp, left, right):
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
		
		if self.op == "**":
				return str("pow(" + stringify(self.left) + ", " + stringify(self.right) + ")")
		return str(stringify(self.left) + " " + self.op + " " + stringify(self.right))



class CBoolOp(CObject):
	
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
		#TODO: What is the C implementation for this?
		if nodeOp == "In":
			self.op = "in"
		self.left = l
		self.right = r

	def stringify(self):
		return str(stringify(self.left) + " " + self.op + " " + stringify(self.right))

class CAssign(CObject):
	
	def __init__(self, t, val):
		self.target = t
		self.val = val

	def stringify(self):
		return str(stringify(self.target) + " = " + stringify(self.val)) + ";"
		# Assignment is always single line.

		#TODO: work out tracking for this (?)

class CForLoop(CObject):

	
	
	def __init__(self, t, i, b):
		self.target = t
		self.iterateOver = i
		self.body = b

	def stringify(self, lists):
		"""Must be called with lists list supplied."""
		iterateOverLPair = get_lPair_from_var(lists, self.iterateOver.var)
		length = iterateOverLPair[1].length
		initialString = "for (int64_t i = 0; " + "i < " + str(length) + "; i++;)"
		bodyString = "{"
		for node in self.body:
			bodyString = bodyString + "\n" + stringify(node)
		bodyString = bodyString + "\n" + "}"
		return initialString + "\n" + bodyString + "\n"

class CWhileLoop(CObject):
	
	def __init__(self, c, b):
		self.cond = c
		self.body = b

	def set_tracker(self, var):
		self.tracker = var

	def stringify(self):
		initialString = "while (" + self.cond.stringify() + ")"
		bodyString = "{"
		for node in self.body:
			bodyString = bodyString + "\n" + stringify(node)
		bodyString = bodyString + "\n" + "}"
		return initialString + "\n" + bodyString + "\n"


class CIf(CObject):
	
	def __init__(self, c, b, f):
		self.cond = c
		self.true = b
		self.false = f

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

def stringify(node):
	if not isinstance(node, CObject):
		return str(node)
	return node.stringify()

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

def get_node_from_var(l, var):
	for node in l:
		if isinstance(node, CVar) and (node.var == var):
			return node
		elif isinstance(node, list):
			return get_node_from_var(node, var)
	return None

def get_lPair_from_var(l, var):
	for node in l:
		if isinstance(node, list) and (node[0].var == var):
			return node




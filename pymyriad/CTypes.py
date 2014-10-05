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

	#TODO: work out how to implement stringify

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
	
	def __init__(self, v, ctx):
		self.var = v
		self.ctx = ctx
		self.attributes = []

	def stringify(self):
		return str(self.var)

class CVarAttr(CVar):
	
	def __init__(self, v, ctx, attr):
		super().__init__(v, ctx)
		self.attr = attr

	#TODO: work out how to implement stringify


class CUnaryOp(CObject):
	
	def __init__(self, op, operand):
		self.op = op
		self.operand = operand
	
	def stringify(self):
		return str(self.op + stringify(self.operand))

class CBinaryOp(CObject):
	
	def __init__(self, op, left, right):
		self.op = op
		self.left = left
		self.right = right

	def stringify(self):
		
		if self.op == "**":
				return str("pow(" + stringify(self.left) + ", " + stringify(self.right) + ")")
		return str(stringify(self.left) + " " + self.op + " " + stringify(self.right))



class CBoolOp(CObject):
	
	def __init__(self, op, vals):
		if op == "Or":
			self.op = "||"
		if op == "And":
			self.op = "&&"
		self.vals = vals

	def stringify(self):
		retString = stringify(self.vals[0])
		for node in self.vals[1:]:
			retString = retString + " " + self.op + " " + stringify(node)
		return retString 

class CCompare(CObject):
	
	def __init__(self, op, l, r):
		if op == "Eq":
			self.op = "=="
		if op == "NotEq":
			self.op = "!="
		if op == "Lt":
			self.op = "<"
		if op == "LtE":
			self.op = "<="
		if op == "Gt":
			self.op = ">"
		if op == "GtE":
			self.op = ">="
		# What is the C implementation for this?
		if op == "In":
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
		return str(stringify(self.target) + " = " + stringify(self.val))
		#TODO: work out tracking for this (?)

class CForLoop(CObject):
	
	def __init__(self, t, i, b):
		self.target = t
		self.iterateOver = i
		self.body = b

	def stringify(self):
		initialString = "for(i = 0; " + "i != " + self.iterateOver.length + "; i++;)"
		bodyString = "{"
		for node in self.body:
			bodyString = bodyString + "\n" + stringify(node)
		bodyString = bodyString + "\n" + "}"
		return initialString + "\n" + bodyString + "\n"

class CWhileLoop(CObject):
	
	def __init__(self, c, b):
		self.cond = c
		self.body = b

	#TODO: work out how to implement stringify


class CIf(CObject):
	
	def __init__(self, c, b, f):
		self.cond = c
		self.true = b
		self.false = f

	def stringify(self):
		initialString = "if (" + stringify(self.cond) + ")"
		bodyString = "{"
		for node in self.body:
			bodyString = bodyString + "\n" + stringify(node)
		bodyString = bodyString + "\n" + "}"
		return initialString + "\n" + bodyString + "\n"

class CReturn(CObject):
	
	def __init__(self, v):
		self.val = v

	def stringify(self):
		return str("return " + stringify(self.val))

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


class CList(object):
	
	def __init__(self, l):
		if not isinstance(l, list):
			raise TypeError("l must be set to a list.")

		self.cargo = l
		self.cargoType = determine_type(l[0])
		self.length = len(l)

class CChar(object):
	
	def __init__(self, c):
		if ((not isinstance(c, str)) or (len(c) != 1)):
			raise TypeError("s must be set to a string of length = 1.")

		self.cargo = c
		self.length = 1
		

class CString(object):
	
	def __init__(self, s):
		if not isinstance(s, str):
			raise TypeError("s must be set to a string.")

		self.cargo = s
		self.length = len(s)

class CVar(object):
	
	def __init__(self, v, ctx):
		self.var = v
		self.ctx = ctx
		self.attributes = []

class CVarAttr(CVar):
	
	def __init__(self, v, ctx, attr):
		super().__init__(v, ctx)
		self.attr = attr

class CUnaryOp(object):
	
	def __init__(self, op, operand):
		self.op = op
		self.operand = operand

class CBinaryOp(object):
	
	def __init__(self, op, left, right):
		self.op = op
		self.left = left
		self.right = right

class CBoolOp(object):
	
	def __init__(self, op, vals):
		if op == "Or":
			self.op = "||"
		if op == "And":
			self.op = "&&"
		self.vals = vals

class CCompare(object):
	
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

class CAssign(object):
	
	def __init__(self, t, val):
		self.target = t
		self.val = val


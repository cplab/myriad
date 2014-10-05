from ast_stringify import *

# Attributes cannot be referenced before their parent object.


class CFunc(object):
	
	def __init__(self, pyCode):
		self.variables = []
		self.nodeList = []
		self.pyCode = pyCode
		
	def parse_python(self):
		parsed = parse(self.pyCode).body
		for node in parsed:
			convertedCType = stringify_expr_contents(node)
			self.nodeList.append(convertedCType)

	

	def track_variables(self, l):

		for node in l:
			if isinstance(node, CVar):
				tempList = []
				for v in self.variables:
					tempList.append(v.var)
				if node.var not in tempList:
					self.variables.append(node)
			elif isinstance(node, CObject):
				self.track_variables(list(node.__dict__.values()))
			elif isinstance(node, list):
				self.track_variables(node)
	

	def track_attributes(self, l):
		for node in l:
			if isinstance(node, CVarAttr):
				target = get_node_from_var(self.variables, node.var)
				if node.attr not in target.attributes:
						target.attributes.append(node.attr)
			elif isinstance(node, CObject):
				self.track_attributes(list(node.__dict__.values()))


def get_node_from_var(l, var):
	for node in l:
		if node.var == var:
			return node
	return None


		

		

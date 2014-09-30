from ast_stringify import *

class CFunc(object):
	
	def __init__(self, pyCode):
		self.variables = []
		self.nodeList = []
		self.pyCode = pyCode
		
	def parse_python(self):
		parsed = parse(self.pyCode).body
		#TODO: keep track of nodes
		# Traverse the list.
		# Inspect every node in case it holds a CVar
		# Resolve from there
		for node in parsed:
			convertedCType = stringify_node(node)
			self.nodeList.append(convertedCType)

	

	def track_variables(self):

		


		for node in self.nodeList:
			print("1")
			if isinstance(node, CVar):
				print("2")
				tempList = []
				for v in self.variables:
					tempList.append(v.var)
					print("3")
				if node.var not in tempList:
					print("Added")
					self.variables.append(node)

		for node in self.nodeList:
			if isinstance(node, CVarAttr):
				for v in self.variables:
					if node.var == v.var and (node.attr not in v.attributes):
						v.attributes.append(node.attr)
													


		
		

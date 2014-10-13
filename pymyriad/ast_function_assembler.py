from ast_stringify import *

# Attributes cannot be referenced before their parent object.
# Lists must be assigned explicitly and reassigned explicitly.


class CFunc(object):
    
    def __init__(self, pyCode):
        self.variables = []
        self.lists = []
        self.nodeList = []
        self.pyCode = pyCode
        
    def parse_python(self):
        parsed = parse(self.pyCode).body
        for node in parsed:
            convertedCType = stringify_node(node)
            self.nodeList.append(convertedCType)

    def get_while_loop_variables(self, l):
        i = 0
        while i < len(self.nodeList):
   
            if isinstance(self.nodeList[i], list):
                get_while_loop_variables(self)
            if isinstance(self.nodeList[i], CForLoop) and isinstance(self.nodeList[i-1], CAssign):
                self.nodeList[i].set_tracker(self.nodeList[i-1].target)
            i = i + 1

    def stringify(self):
        retString = ""
        for node in self.nodeList:
            print(node.stringify())
            retString = retString + node.stringify() + "\n"
        print("---")
        print(retString)

            

    

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

    #TODO: write tie_variables(self, l) which will tie variables to their initial values.
    # Easily done because all variables are the first instance of themselves.
    # Just look at their CAssign statment container.z
    

    def track_attributes(self, l):
        for node in l:
            if isinstance(node, CVarAttr):
                target = get_node_from_var(self.variables, node.var)
                if node.attr not in target.attributes:
                        target.attributes.append(node.attr)
            elif isinstance(node, CObject):
                self.track_attributes(list(node.__dict__.values()))

    def track_lists(self, l):
        #TODO: rerun this when updating any lists. Erase self.lists and repopulate.
        for node in l:
            if isinstance(node, CAssign) and isinstance(node.val, CList):
                self.lists.append([node.target, node.val])
            elif isinstance(node,CObject):
                self.track_variables(list(node.__dict__.values()))
            elif isinstance(node, list):
                self.track_variables(node)


    def tie_lists(self, variables, lists):
        
        # Needs to only get first list values so that we have a source for the list declaration.
        for lPair in lists:
            for node in variables:
                if node.var == lPair[0].var:
                    lPair[0] = node
            
                
                


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



        

        

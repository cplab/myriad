"""
Tester module for CTypes and ast_parse.
Author: Alex Davies
"""

import ast_parse
import ast
import unittest


class CTypesTester(unittest.TestCase):

    def test_CList(self):
        code = ast.parse("[1, 2, 3]")
        parsed = ast_parse.parse_literal(code.body[0].value)
        self.assertEqual(parsed.cargo, [1, 2, 3])
        self.assertEqual(parsed.length, 3)
        self.assertEqual(parsed.cargoType, "int")
        self.assertEqual(parsed.stringify(), "{1, 2, 3}")

        code = ast.parse("[1, 2, [6, 7], [8, 9]]")
        parsed = ast_parse.parse_literal(code.body[0].value)
        self.assertEqual(parsed.length, 4)
        self.assertEqual(parsed.stringify(), "{1, 2, {6, 7}, {8, 9}}")

    def test_CSubscript(self):
        code = ast.parse("l[0]")
        parsed = ast_parse.parse_subscript(code.body[0].value)
        self.assertEqual(parsed.val, "l")
        self.assertEqual(parsed.sliceDict["Index"], 0)
        self.assertEqual(parsed.stringify(), "l[0]")

    def test_CVar(self):
        code = ast.parse("a")
        parsed = ast_parse.parse_var(code.body[0].value)
        self.assertEqual(parsed.var, "a")
        self.assertEqual(parsed.ctx, "Load")
        self.assertEqual(parsed.stringify(), "a")

    def test_CVarAttr(self):
        code = ast.parse("a.foo")
        parsed = ast_parse.parse_attribute(code.body[0].value)
        self.assertEqual(parsed.var.var, "a")
        self.assertEqual(parsed.attr, "foo")
        code = ast.parse("a.foo.bar")
        # TODO: parsed = stringify_attribute(code.body)

    def test_CUnaryOp(self):
	    code = ast.parse("-a")
	    parsed = ast_parse.parse_unaryop(code.body[0].value)
	    self.assertEqual(parsed.op, "-")
	    self.assertEqual(parsed.operand.var, "a")
	    self.assertEqual(parsed.stringify(), "-a")

    def test_CBinaryOp(self):
        code = ast.parse("1 + 2")
        parsed = ast_parse.parse_binaryop(code.body[0].value)
        self.assertEqual(parsed.left, 1)
        self.assertEqual(parsed.right, 2)
        self.assertEqual(parsed.op, "+")
        self.assertEqual(parsed.stringify(), "1 + 2")

        code = ast.parse("a ** b")
        parsed = ast_parse.parse_binaryop(code.body[0].value)
        self.assertEqual(parsed.left.var, "a")
        self.assertEqual(parsed.right.var, "b")
        self.assertEqual(parsed.op, "**")
        self.assertEqual(parsed.stringify(), "pow(a, b)")

    def test_CBoolOp(self):
        code = ast.parse("a and b")
        parsed = ast_parse.parse_boolop(code.body[0].value)
        self.assertEqual(parsed.vals[0].var, "a")
        self.assertEqual(parsed.vals[1].var, "b")
        self.assertEqual(parsed.op, "&&")
        self.assertEqual(parsed.stringify(), "a && b")

        code = ast.parse("a and b and c")
        parsed = ast_parse.parse_boolop(code.body[0].value)
        self.assertEqual(parsed.vals[0].var, "a")
        self.assertEqual(parsed.vals[1].var, "b")
        self.assertEqual(parsed.vals[2].var, "c")
        self.assertEqual(parsed.op, "&&")
        self.assertEqual(parsed.stringify(), "a && b && c")

    def test_CCompare(self):
        code = ast.parse("a <= 10")
        parsed = ast_parse.parse_compare(code.body[0].value)
        self.assertEqual(parsed.op, "<=")
        self.assertEqual(parsed.left.var, "a")
        self.assertEqual(parsed.right, 10)
        self.assertEqual(parsed.stringify(), "a <= 10")

    def test_CAssign(self):
        code = ast.parse("a = 5")
        parsed = ast_parse.parse_assign(code.body[0])
        self.assertEqual(parsed.target.var, "a")
        self.assertEqual(parsed.val, 5)
        self.assertEqual(parsed.stringify(), "a = 5;")

        code = ast.parse("a = b")
        parsed = ast_parse.parse_assign(code.body[0])
        self.assertEqual(parsed.target.var, "a")
        self.assertEqual(parsed.val.var, "b")
        self.assertEqual(parsed.stringify(), "a = b;")

        code = ast.parse("a = [1, 2, 3]")
        parsed = ast_parse.parse_assign(code.body[0])
        self.assertEqual(parsed.target.var, "a")
        self.assertEqual(parsed.val.cargo, [1, 2, 3])
        self.assertEqual(parsed.stringify(), "a = {1, 2, 3};")

    def test_CForLoop(self):
        # Easier to test individually/in a CFunc tester due to use of lists
        pass        

    def test_CWhileLoop(self):
        code = ast.parse("while x != 4:\n    y = y + 1")
        parsed = ast_parse.parse_while_loop(code.body[0])
        expected = "while (x != 4)\n{\ny = y + 1;\n}\n"
        self.assertEqual(parsed.stringify(), expected)

    def test_CIf(self):
        code = ast.parse("if a == b:\n    x = x + 1\nelse:\n    x = x - 1")
        parsed = ast_parse.parse_if_statement(code.body[0])
        expected = "if (a == b)\n{\nx = x + 1;\n}else{\nx = x + 1;\n"
        print("\n")
        print("Actual:")
        print(parsed.stringify())
        print("\n")
        print("Expected:")
        print(expected)    
           
                

if __name__ == '__main__':
    unittest.main()

from CTypes import *
from ast_stringify import *
import ast
import unittest


class CTypesTester(unittest.TestCase):

        def test_CList(self):
                code = parse("[1, 2, 3]")
                parsed = stringify_literal(code.body[0].value)
                self.assertEqual(parsed.cargo, [1, 2, 3])
                self.assertEqual(parsed.length, 3)
                self.assertEqual(parsed.cargoType, "int")
                self.assertEqual(parsed.stringify(), "{1, 2, 3}")

                code = parse("[1, 2, [6, 7], [8, 9]]")
                parsed = stringify_literal(code.body[0].value)
                self.assertEqual(parsed.length, 4)
                self.assertEqual(parsed.stringify(), "{1, 2, {6, 7}, {8, 9}}")

        def test_CSubscript(self):
                code = parse("l[0]")
                parsed = stringify_subscript(code.body[0].value)
                self.assertEqual(parsed.val, "l")
                self.assertEqual(parsed.sliceDict["Index"], 0)
                self.assertEqual(parsed.stringify(), "l[0]")

        def test_CVar(self):
                code = parse("a")
                parsed = stringify_var(code.body[0].value)
                self.assertEqual(parsed.var, "a")
                self.assertEqual(parsed.ctx, "Load")
                self.assertEqual(parsed.stringify(), "a")

        def test_CVarAttr(self):
                code = parse("a.foo")
                parsed = stringify_attribute(code.body[0].value)
                self.assertEqual(parsed.var, "a")
                self.assertEqual(parsed.attr, "foo")
                code = parse("a.foo.bar")
                # TODO: parsed = stringify_attribute(code.body


if __name__ == '__main__':
        unittest.main()

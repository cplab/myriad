#!/usr/bin/python3
from enum import Enum as PyEnum
import copy

from pycparser import parse_file, c_generator
from pycparser.c_ast import *

from m_annotations import enforce_annotations

"""
  Typedef: ctor_t, [], ['typedef']
    PtrDecl: []
      FuncDecl:
        ParamList:
          Decl: self, [], [], []
            PtrDecl: []
              TypeDecl: self, []
                IdentifierType: ['void']
          Decl: app, [], [], []
            PtrDecl: []
              TypeDecl: app, []
                IdentifierType: ['va_list']
        PtrDecl: []
          TypeDecl: ctor_t, []
            IdentifierType: ['void']
"""


class MyriadCType(PyEnum):
    m_float = IdentifierType(names=["float"])
    m_double = IdentifierType(names=["double"])
    m_int = IdentifierType(names=["int"])
    m_uint = IdentifierType(names=["unsigned int"])
    m_void = IdentifierType(names=["void"])
    m_va_list = IdentifierType(names=["va_list"])
#    m_struct = IdentifierType(names=["struct"])


class MyriadScalar(object):
    """
    Object for representing any individual C scalar variable.
    """
    _cgen = c_generator.CGenerator()

    @enforce_annotations
    def __init__(self, ident: str, base_type: MyriadCType, ptr: bool=False):
        self.ident = ident
        self.base_type = base_type
        self.ptr = ptr

        # Initialize internal C type declaration
        self.type_decl = TypeDecl(declname=self.ident,
                                  quals=[],
                                  type=self.base_type.value)

        # Initialize internal C ptr declaration (might not be used)
        self.ptr_decl = PtrDecl(quals=[], type=self.type_decl)

        # Initialize internal top-level declaration
        self.decl = Decl(name=self.ident,
                         quals=[],
                         storage=[],
                         funcspec=[],
                         type=self.ptr_decl if ptr else self.type_decl,
                         init=None,
                         bitsize=None)

    @enforce_annotations
    def stringify_decl(self) -> str:
        return self._cgen.visit(self.decl)

    @enforce_annotations
    def stringify_type_decl(self) -> str:
        if self.ptr:
            return self._cgen_visit(self.ptr_decl)
        else:
            return self._cgen_visit(self.type_decl)


class MyriadFunction(object):

    _cgen = c_generator.CGenerator()

    @enforce_annotations
    def __init__(self,
                 ident: str,
                 args_list: list=[],
                 ret_var: MyriadScalar=None):
        # Make sure we got the right parameter types
        assert(all(type(elem) is MyriadScalar for elem in args_list))

        self.ident = ident

        # If no return value is given, assume void
        self.ret_var = ret_var
        if self.ret_var is None:
            self.ret_var = MyriadScalar(self.ident, MyriadCType.m_void)

        # Args list stores the MyriadScalars
        self.args_list = args_list
        # Param list stores the scalar's declarations
        self.param_list = ParamList([v.decl for v in self.args_list])

        # ------------------------------------------
        # Create internal c_ast function declaration
        # ------------------------------------------

        _tmp_decl = copy.deepcopy(self.ret_var.decl.type)
        # Make sure we override the identifier in our copy
        if type(_tmp_decl) is PtrDecl:
            _tmp_decl.type.declname = self.ident
        else:
            _tmp_decl.declname = self.ident

        self.func_decl = FuncDecl(self.param_list, _tmp_decl)
        self.decl = Decl(name=self.ident,
                         quals=[],
                         storage=[],
                         funcspec=[],
                         type=self.func_decl,
                         init=None,
                         bitsize=None)

        # -----------------------------------------
        # TODO: Create internal c_ast function definition
        # -----------------------------------------
        self.func_def = None

    @enforce_annotations
    def stringify_decl(self) -> str:
        return self._cgen.visit(self.decl)


def test_ast():
    filename = 'test.c'
    ast = parse_file(filename,
                     use_cpp=True,
                     cpp_path='gcc',
                     cpp_args=['-E', r'-I../utils/fake_libc_include'])

    ast.show()

    cgen = c_generator.CGenerator()

    with open("test.c", 'w', encoding="utf-8") as f:
        for node in ast.ext:
            f.write(cgen.visit(node) + ";\n")

    void_id = IdentifierType(names=["void"])
    self_arg = Decl("self", [], [], [],
                    PtrDecl([], TypeDecl("self", [], void_id)), None, None)

    print(cgen.visit(self_arg))
    print(cgen.visit(ast.ext[121].type.type.args.params[0]))


def main():
    # Test Scalar
    m = MyriadScalar("self", MyriadCType.m_void, True)
    print(m.stringify_decl())

    f = MyriadFunction("myriad_dtor", [m])
    print(f.stringify_decl())


if __name__ == "__main__":
    main()

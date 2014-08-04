#!/usr/bin/python3
"""
TODO: Docstring
"""
from enum import Enum as PyEnum
import copy

from pycparser import parse_file, c_generator
from pycparser.c_ast import *

from m_annotations import enforce_annotations


class _MyriadBase(object):
    """Common core class for Myriad types"""
    _cgen = c_generator.CGenerator()

    def __init__(self, ident, decl=None):
        # TODO: Namespace collision detection?
        self.ident = ident
        self.decl = decl

    @enforce_annotations
    def stringify_decl(self) -> str:
        """Renders the internal C declaration"""
        return self._cgen.visit(self.decl)


class MyriadCType(PyEnum):
    """Base C types available for scalars."""
    m_float = IdentifierType(names=["float"])
    m_double = IdentifierType(names=["double"])
    m_int = IdentifierType(names=["int"])
    m_uint = IdentifierType(names=["unsigned int"])
    m_void = IdentifierType(names=["void"])
    m_va_list = IdentifierType(names=["va_list"])
    m_struct = IdentifierType(names=["struct"])


class MyriadScalar(_MyriadBase):
    """Object for representing any individual C scalar variable."""

    @enforce_annotations
    def __init__(self, ident: str, base_type: MyriadCType, ptr: bool=False):
        # Always call super first
        super().__init__(ident)

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


class MyriadStruct(_MyriadBase):

    @enforce_annotations
    def __init__(self, ident: str, struct_name: str, members: list=[]):

        # Make sure we got the right parameter types
        if not all(issubclass(m.__class__, _MyriadBase) for m in members):
            raise TypeError("Invalid struct member(s) type(s).")

        self.ident = ident
        self.struct_name = struct_name
        self.base_type = MyriadCType.m_struct

        # Set struct members
        self.members = [v.decl for v in members]

        _tmp_decl = Decl(name=self.ident,
                         quals=[],
                         storage=[],
                         funcspec=[],
                         type=Struct(self.struct_name, self.members),
                         init=None,
                         bitsize=None)

        # Need to call super last in this instance
        super().__init__(ident, _tmp_decl)


class MyriadFunction(_MyriadBase):
    """Function container for Myriad functions"""

    @enforce_annotations
    def __init__(self,
                 ident: str,
                 args_list: list=[],
                 ret_var: MyriadScalar=None,
                 typedef_name: str=None,
                 gen_typedef: bool=False):
        # Always call super first
        super().__init__(ident)

        # --------------------------------------------
        # Set return value; none is given, assume void
        # --------------------------------------------
        self.ret_var = ret_var
        if self.ret_var is None:
            self.ret_var = MyriadScalar(self.ident, MyriadCType.m_void)

        # --------------------------------------------
        #  Create internal representation of args list
        # --------------------------------------------

        # Make sure we got the right parameter types
        if not all(issubclass(e.__class__, _MyriadBase) for e in args_list):
            raise TypeError("Invalid function argument(s) type(s).")

        self.args_list = args_list # Args list stores the MyriadScalars

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

        # ------------------------------------
        # Create internal typedef, if specfied
        # ------------------------------------
        self.fun_typedef = None

        if gen_typedef:
            if typedef_name is None:
                typedef_name = self.ident + "_t"
            self.gen_typedef(typedef_name)

        # -----------------------------------------------
        # TODO: Create internal c_ast function definition
        # -----------------------------------------------
        self.func_def = None

    @enforce_annotations
    def gen_typedef(self, typedef_name: str):
        """Generates a typedef definition with the given name."""

        _tmp = IdentifierType(names=self.func_decl.type.type.names)
        tmp = PtrDecl([], TypeDecl(typedef_name, [], _tmp))
        _tmp_fdecl = PtrDecl([], FuncDecl(self.param_list, tmp))

        self.fun_typedef = Typedef(name=typedef_name,
                                   quals=[],
                                   storage=['typedef'],
                                   type=_tmp_fdecl,
                                   coord=None)

    @enforce_annotations
    def stringify_typedef(self) -> str:
        return self._cgen.visit(self.fun_typedef)


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
    void_ptr = MyriadScalar("self", MyriadCType.m_void, True)
    print(void_ptr.stringify_decl())
    # Test Function
    myriad_dtor = MyriadFunction("myriad_dtor", [void_ptr], gen_typedef=True)
    print(myriad_dtor.stringify_decl())
    print(myriad_dtor.stringify_typedef())
    # Test struct
    myriad_class = MyriadStruct(None, "MyriadClass", [void_ptr])
    print(myriad_class.stringify_decl())


if __name__ == "__main__":
    main()

#!/usr/bin/python3
"""
TODO: Docstring
"""

from enum import Enum as PyEnum
from enum import unique
import copy

from pycparser import parse_file, c_generator
from pycparser.c_ast import IdentifierType, Typedef
from pycparser.c_ast import Decl, PtrDecl, TypeDecl
from pycparser.c_ast import Struct, FuncDecl
from pycparser.c_ast import ParamList

from myriad_utils import enforce_annotations

# Global TODOs
# TODO: add support for __hash__ and __eq__ for set purposes


class _MyriadBase(object):
    """Common core class for Myriad types"""
    _cgen = c_generator.CGenerator()

    def __init__(self,
                 ident,
                 decl=None,
                 quals: list=None,
                 storage: list=None):
        self.ident = ident
        self.decl = decl
        # TODO: check valid quals, storage types
        self.quals = [] if quals is None else quals
        self.storage = [] if storage is None else storage

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
    def __init__(self,
                 ident: str,
                 base_type: MyriadCType,
                 ptr: bool=False,
                 quals: list=None,
                 storage: list=None):
        # Always call super first
        super().__init__(ident, quals=quals, storage=storage)

        self.base_type = base_type
        self.ptr = ptr

        # Initialize internal C type declaration
        self.type_decl = TypeDecl(declname=self.ident,
                                  quals=self.quals,
                                  type=self.base_type.value)

        # Initialize internal C ptr declaration (might not be used)
        self.ptr_decl = PtrDecl(quals=[], type=self.type_decl)

        # Initialize internal top-level declaration
        self.decl = Decl(name=self.ident,
                         quals=self.quals,
                         storage=self.storage,
                         funcspec=[],
                         type=self.ptr_decl if ptr else self.type_decl,
                         init=None,
                         bitsize=None)


class MyriadStructType(_MyriadBase):
    """Struct construct"""

    @enforce_annotations
    def __init__(self,
                 struct_name: str,
                 members: set={},
                 storage: list=None):

        # Make sure we got the right parameter types
        if not all(issubclass(m.__class__, _MyriadBase) for m in members):
            raise TypeError("Invalid struct member(s) type(s).")

        self.struct_name = struct_name
        self.base_type = MyriadCType.m_struct

        # Set struct members
        # TODO: Struct members must be unique
        self.members = {v.decl for v in members}

        _tmp_decl = Decl(name=None,
                         quals=[],
                         storage=[],
                         funcspec=[],
                         type=Struct(self.struct_name, list(self.members)),
                         init=None,
                         bitsize=None)

        # Need to call super last in this instance
        super().__init__(None, _tmp_decl, storage=storage)

    class MyriadStructVar(MyriadScalar):
        """ Actual struct variable instance. """

        # TODO: Provide a robust mechanism for initialization of members.
        # Most likely this would involve enforcing kwargs variables to be of
        # the form (member_name, init_value) where init_value may be set to
        # None to indicate no initialization (NULL is just 0). Probably would
        # be a good idea to seperate out initialization of the struct members
        # into a seperate function. Initial values for members should NOT be
        # supported for pointer types. Members should be attributes.
        def __init__(self,
                     prototype: MyriadStructType,  # NOQA
                     ident: str,
                     ptr: bool=False,
                     quals: list=None,
                     storage: list=None,
                     **kwargs):
            """ TODO """
            # TODO: Eventually call super()

            self.prototype = prototype  # TODO: Do other things with this?

            # Initialize members as attributes
            # TODO: do actual initialization of mmebers values elsewhere
            for member_name in kwargs.keys():
                self.__dict__[member_name] = None

    def __call__(self):
        # TODO: Return an instance of MyriadStructVar w/ arg passthrough
        pass


@unique
class MyriadFunType(PyEnum):
    """ Enumerator for different function types """
    m_delegator = 1
    m_module = 2
    m_method = 3


# pylint: disable=R0902
class MyriadFunction(_MyriadBase):
    """Function container for Myriad functions"""

    @enforce_annotations
    def __init__(self,
                 ident: str,
                 args_list: set=None,
                 ret_var: MyriadScalar=None,
                 fun_type: MyriadFunType=MyriadFunType.m_module):
        # Always call super first
        super().__init__(ident)

        # --------------------------------------------
        # Set return value; none is given, assume void
        # --------------------------------------------
        self.ret_var = ret_var
        if self.ret_var is None:
            self.ret_var = MyriadScalar(self.ident, MyriadCType.m_void)

        # -----------------------------------------------------
        # Set function type/scope: module, method, or delegator
        # -----------------------------------------------------
        self.fun_type = fun_type

        # --------------------------------------------
        #  Create internal representation of args list
        # --------------------------------------------

        # Make sure we got the right parameter types
        if not all(issubclass(e.__class__, _MyriadBase) for e in args_list):
            raise TypeError("Invalid function argument(s) type(s).")

        # Args list stores the MyriadScalars
        self.args_list = [] if args_list is None else list(args_list)

        # Param list stores the scalar's declarations in a C AST object.
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

        # -------------------------------------------
        # Generate typedef depending on function type
        # -------------------------------------------
        self.fun_typedef = None
        if self.fun_type is MyriadFunType.m_method:
            self.gen_typedef()

        # -----------------------------------------------
        # TODO: Create internal c_ast function definition
        # -----------------------------------------------
        self.func_def = None

    @enforce_annotations
    def gen_typedef(self, typedef_name: str=None):
        """Generates a typedef definition for this function."""

        if typedef_name is None:
            typedef_name = self.ident + "_t"

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
        """Returns string representation of this function's typedef"""
        return self._cgen.visit(self.fun_typedef)


def test_ast():
    """Inspect C soure code using AST and re-render"""
    filename = 'test.c'
    ast = parse_file(filename,
                     use_cpp=True,
                     cpp_path='gcc',
                     cpp_args=['-E', r'-I../utils/fake_libc_include'])

    ast.show()

    cgen = c_generator.CGenerator()

    with open("test.c", 'w', encoding="utf-8") as test_file:
        for node in ast.ext:
            test_file.write(cgen.visit(node) + ";\n")

    void_id = IdentifierType(names=["void"])
    self_arg = Decl("self", [], [], [],
                    PtrDecl([], TypeDecl("self", [], void_id)), None, None)

    print(cgen.visit(self_arg))
    print(cgen.visit(ast.ext[121].type.type.args.params[0]))


def main():
    """Test basic functionality"""
    # Test Scalar
    void_ptr = MyriadScalar("self", MyriadCType.m_void, True, quals=["const"])
    print(void_ptr.stringify_decl())
    # Test Function
    myriad_dtor = MyriadFunction("myriad_dtor", {void_ptr})
    myriad_dtor.gen_typedef()
    print(myriad_dtor.stringify_decl())
    print(myriad_dtor.stringify_typedef())
    # Test struct
    myriad_class = MyriadStructType("MyriadClass", {void_ptr})
    print(myriad_class.stringify_decl())


if __name__ == "__main__":
    main()

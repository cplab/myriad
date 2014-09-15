#!/usr/bin/python3
"""
TODO: Docstring
"""

from enum import Enum as PyEnum
from enum import unique
from collections import OrderedDict
import copy

from pycparser import parse_file, c_generator
from pycparser.c_ast import IdentifierType, Typedef
from pycparser.c_ast import Decl, PtrDecl, TypeDecl, ID
from pycparser.c_ast import Struct, FuncDecl
from pycparser.c_ast import ParamList

from myriad_utils import enforce_annotations, assert_list_type

# Global TODOs
# TODO: add support for __hash__ and __eq__ for dict/set purposes


class _MyriadBase(object):
    """ Common core class for Myriad types. """
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


# ----------------------------------------
#                 CTYPES
# ----------------------------------------

MyriadCType = type("MyriadCType", (object,), {'mtype': None})
MFloat = type("MFloat",
              (MyriadCType,),
              {
                  'mtype': IdentifierType(names=["float"]),
              })()
MDouble = type("MDouble",
               (MyriadCType,),
               {
                   'mtype': IdentifierType(names=["double"]),
               })()
MInt = type("MInt",
            (MyriadCType,),
            {
                'mtype': IdentifierType(names=["int64_t"]),
            })()
MUInt = type("MUInt",
             (MyriadCType,),
             {
                 'mtype': IdentifierType(names=["uint64_t"]),
             })()
MVoid = type("MVoid",
             (MyriadCType,),
             {
                 'mtype': IdentifierType(names=["void"]),
             })()
MSizeT = type("MSizeT",
              (MyriadCType,),
              {
                  'mtype': IdentifierType(names=["size_t"]),
              })()
MVarArgs = type("MVarArgs",
                (MyriadCType,),
                {
                    'mtype': IdentifierType(names=["va_list"]),
                })()


class MyriadScalar(_MyriadBase):
    """Object for representing any individual C scalar variable."""

    # pylint: disable=R0913
    @enforce_annotations
    def __init__(self,
                 ident: str,
                 base_type: MyriadCType,
                 ptr: bool=False,
                 quals: list=None,
                 storage: list=None,
                 init=None):
        # Always call super first
        super().__init__(ident, quals=quals, storage=storage)

        self.base_type = base_type
        self.ptr = ptr

        # Initialize internal C type declaration
        self.type_decl = TypeDecl(declname=self.ident,
                                  quals=self.quals,
                                  type=self.base_type.mtype)

        # Initialize internal C ptr declaration (might not be used)
        self.ptr_decl = PtrDecl(quals=[], type=self.type_decl)

        # Initialize internal top-level declaration
        self.decl = Decl(name=self.ident,
                         quals=self.quals,
                         storage=self.storage,
                         funcspec=[],
                         type=self.ptr_decl if ptr else self.type_decl,
                         init=init,  # TODO: Process init
                         bitsize=None)

    @enforce_annotations
    def typematch(self, other) -> bool:
        """ Checks if other is of an equivalent type. """

        if other is None or not issubclass(other, _MyriadBase):
            return False

        if issubclass(other, MyriadScalar):
            return other.base_type is self.base_type
        else:  # TODO: Figure out what to do when given non-scalar type
            pass


class MyriadStructType(_MyriadBase):
    """Struct construct"""

    @enforce_annotations
    def __init__(self,
                 struct_name: str,
                 members: OrderedDict=None,
                 storage: list=None):

        # Make sure we got the right parameter types
        assert_list_type(list(members.values()), _MyriadBase)

        self.struct_name = struct_name

        # Set struct members using ordered dict: order matters for memory!
        members = OrderedDict() if members is None else members
        sorted_members = [members[idx].decl for idx in sorted(members.keys())]
        members = {v.ident: v.decl for v in members.values()}
        self.members = OrderedDict()
        for member_ident, member in members.items():
            self.members[member_ident] = member

        # Set struct type
        self.struct_c_ast = Struct(self.struct_name, sorted_members)

        # Setup instance generator (i.e. "the factory" class)
        self.base_type = type(self.struct_name,
                              (MyriadCType,),
                              {'mtype': Struct(self.struct_name, None)})()

        _tmp_decl = Decl(name=None,
                         quals=[],
                         storage=[],
                         funcspec=[],
                         type=self.struct_c_ast,
                         init=None,
                         bitsize=None)

        # Need to call super last in this instance
        super().__init__(None, _tmp_decl, storage=storage)

    def __call__(self,
                 ident: str,
                 ptr: bool=False,
                 quals: list=None,
                 storage: list=None,
                 init=None,
                 **kwargs):
        """ Factory method for making struct instances from template. """

        # Assert we receive at least a subset of arguments
        prot_idents = set(self.members.keys())
        given_idents = set(kwargs.keys())
        if not given_idents.issubset(prot_idents):
            diff = given_idents - prot_idents
            msg = "{0} not valid member(s) of struct {1}"
            raise NameError(msg.format(str(diff), self.ident))

        # TODO: Do more robust checking with namespace lookups
        # Assert argument initial values match protoype's
        new_kwargs = {}
        for arg_id, arg_val in kwargs.items():
            if issubclass(arg_val, MyriadScalar):
                # Check if underlying types match
                if self.members[arg_id].typematch(arg_val) is False:
                    msg = "Member {0} is of type {1}, argument is of type {2}."
                    raise TypeError(msg.format(arg_id,
                                               self.members[arg_id].base_type,
                                               arg_val))
                else:
                    # If a scalar, pass forward as ID
                    new_kwargs[arg_id] = ID(arg_val.ident)

        # CONSTRUCTOR GOES HERE
        new_instance = MyriadScalar(ident, self.base_type, ptr,
                                    quals, storage, init)

        # TODO: Actually initialize versus just setting blindly
        # Initialize members as attributes
        for member_name, member_val in new_kwargs.items():
            new_instance.__dict__[member_name] = member_val

        return new_instance


@unique
class MyriadFunType(PyEnum):
    """ Enumerator for different function types """
    m_delg = 1
    m_module = 2
    m_method = 3


# pylint: disable=R0902
class MyriadFunction(_MyriadBase):
    """Function container for Myriad functions"""

    @enforce_annotations
    def __init__(self,
                 ident: str,
                 args_list: OrderedDict=None,
                 ret_var: MyriadScalar=None,
                 fun_type: MyriadFunType=MyriadFunType.m_module,
                 fun_def=None):
        # Always call super first
        super().__init__(ident)

        # --------------------------------------------
        # Set return value; none is given, assume void
        # --------------------------------------------
        self.ret_var = ret_var
        if self.ret_var is None:
            self.ret_var = MyriadScalar(self.ident, MVoid)

        # -----------------------------------------------------
        # Set function type/scope: module, method, or delegator
        # -----------------------------------------------------
        self.fun_type = fun_type

        # --------------------------------------------
        #  Create internal representation of args list
        # --------------------------------------------

        # Make sure we got the right parameter types
        assert_list_type(list(args_list.values()), _MyriadBase)

        # Args list stores the MyriadScalars as ordered parameters
        self.args_list = OrderedDict() if args_list is None else args_list

        # Param list stores the scalar's declarations in a C AST object.
        self.param_list = ParamList([v.decl for v in self.args_list.values()])

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
        self.base_type = None
        self.gen_typedef()

        # -----------------------------------------------
        # TODO: Create internal c_ast function definition
        # -----------------------------------------------
        self.fun_def = fun_def

    @enforce_annotations
    def gen_typedef(self, typedef_name: str=None):
        """Generates a typedef definition for this function."""

        # TODO: Do we need to prevent double generation?

        if typedef_name is None:
            typedef_name = self.ident + "_t"

        _tmp = None
        if type(self.func_decl.type.type) is TypeDecl:
            _tmp = self.func_decl.type.type.type
        else:
            _tmp = IdentifierType(names=self.func_decl.type.type.names)
        tmp = PtrDecl([], TypeDecl(typedef_name, [], _tmp))
        _tmp_fdecl = PtrDecl([], FuncDecl(self.param_list, tmp))

        self.base_type = type(typedef_name,
                              (MyriadCType,),
                              {'mtype': IdentifierType([typedef_name])})()

        self.fun_typedef = Typedef(name=typedef_name,
                                   quals=[],
                                   storage=['typedef'],
                                   type=_tmp_fdecl,
                                   coord=None)

    def stringify_typedef(self) -> str:
        """ Returns string representation of this function's typedef. """
        return self._cgen.visit(self.fun_typedef)

    def stringify_def(self) -> str:
        """ Returns string representation of this function's definition. """
        if type(self.fun_def) is str:
            return self.fun_def
        else:
            raise NotImplementedError("Non-string representations unsupported")


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
    void_ptr = MyriadScalar("self", MVoid, True, quals=["const"])
    print(void_ptr.stringify_decl())
    # Test Function
    myriad_dtor = MyriadFunction("myriad_dtor", OrderedDict({0: void_ptr}))
    myriad_dtor.gen_typedef()
    print(myriad_dtor.stringify_decl())
    print(myriad_dtor.stringify_typedef())
    # Test struct
    myriad_class = MyriadStructType("MyriadClass", OrderedDict({0: void_ptr}))
    print(myriad_class.stringify_decl())
    class_m = myriad_class("class_m", quals=["const"])
    print(class_m.stringify_decl())


if __name__ == "__main__":
    main()

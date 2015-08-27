"""
.. module:: myriad_types
   :platform: Unix, Windows, Mac OS X
   :synopsis: Provides the underlying C-API type system interface.

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>

Purpose of this module is to allow for in-memory variables that contain
C type information, presented as mutable Python variables. C code can be
generated using the various stringify_* functions.

Run-time type information is done through extending the 'abstract',
programmatically-generated toplevel MyriadCType class. Struct types and
function typedefs are both dynamically generated to subclass this type
to allow for robust run-time type checking. A set of default MyriadCType
subclasses is provided for normal use (e.g. MVoid, MInt, etc.)
"""

from collections import OrderedDict
import copy
import inspect
from warnings import warn

from pycparser import c_generator
from pycparser.c_ast import IdentifierType, Typedef
from pycparser.c_ast import Decl, PtrDecl, TypeDecl, ID, ArrayDecl
from pycparser.c_ast import Struct, FuncDecl
from pycparser.c_ast import ParamList

from myriad_utils import enforce_annotations, assert_list_type


# TODO: add support for __hash__ and __eq__ for dict/set purposes
class _MyriadBase(object):
    """
    Common core class for Myriad types.

    All myriad types extend from this class, since it provides the most
    basic of all C features: declarations. Note that declarations are purely
    of the form "<qualifiers> <type> <declaration name>" and do not include
    initialization information by default.
    """

    #: Class-level c generator using pycparser's C AST traversal tool.
    _cgen = c_generator.CGenerator()

    def __init__(self,
                 ident: str,
                 decl: Decl=None,
                 quals: list=None,
                 storage: list=None):
        """
        Initializes a base type skeleton object.

        :param str ident: Name of the object to be created.
        :param Decl decl: C AST declaration used by pycparser to generate code.
        :param list quals: C AST scope qualifiers (e.g. "static")
        :param list storage: C AST storage qualifiers (e.g. "const")
        """

        #: String identifier at the C level.
        self.ident = ident

        #: pycparser's C ast declaration node.
        self.decl = decl

        #: C scope qualifiers for the declaration (e.g. "static").
        self.quals = [] if quals is None else quals

        #: C storage qualifiers for the declaration (e.g. "const").
        self.storage = [] if storage is None else storage

    def stringify_decl(self) -> str:
        """ Renders the internal C declaration and returns it as a string. """
        return self._cgen.visit(self.decl)


_C_DECL_MAPPING = {
    "int": 'i',
    "char": 'b',
    "short int": 'h',
    "long int": 'l',
    "unsigned char": 'B',
    "unsigned int": 'I',
    "unsigned long": 'k',
    "int64_t": 'L',
    "uint64_t": 'K',
    "size_t": 'n',
    "float": 'f',
    "double": 'd',
}


def c_decl_to_pybuildarg(c_decl: Decl):
    """Returns the Py_BuildValue character associated with the declaration."""
    if c_decl is None:
        raise TypeError("Argument cannot be None")
    return _C_DECL_MAPPING.get(c_decl.type.type.names[0], d=None)

# ----------------------------------------
#                 CTYPES
# ----------------------------------------

# Top-level C types are programmatically-generated classes that are used
# for providing IdentifierType information in a "Pythonic" way. The advantage
# is that these declarations are all childern of MyriadCType, allowing for
# run-time typechecking for valid C types and extension of the type system
# via subclassing.
MyriadCType = type("MyriadCType", (object,), {"mtype": None})

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
    """
    Object for representing any individual C scalar variable.

    Represents a 'scalar', meaning any individual variable declaration,
    complete with information about its scope and storage qualifiers.
    Enforcing having to pass a valid MyriadCType means that only scalars of
    pre-registered types (e.g. MInt) can be instantiated, allowing for run-time
    and parse-time type checking.
    """

    @enforce_annotations
    def __init__(self,
                 ident: str,
                 base_type: MyriadCType,
                 ptr: bool=False,
                 quals: list=None,
                 storage: list=None,
                 arr_id: str=None,
                 init=None):
        """
        Initializes a scalar variable, i.e. a base type, a pointer type, or an
        array type containing the base scalar type (variable/fixed length).

        :param str ident: Name of the scalar variable to be created.
        :param MyriadCType base_type: Underlying C AST base type (e.g. MInt).
        :param bool ptr: Indicates whether this is a pointer.
        :param list quals: C AST scope qualifiers (e.g. "static")
        :param list storage: C AST storage qualifiers (e.g. "const")
        :param str arr_id: Array length ID specifier (None if this isn't one)
        :param init: Initial value given to this scalar (TODO: NOT IMPLEMENTED)
        """
        # Always call super first
        super().__init__(ident, quals=quals, storage=storage)

        #: Represents the underlying C type via a MyriadCType subclass.
        self.base_type = base_type

        #: Indicates if this is a pointer (or an array of pointers).
        self.ptr = ptr

        #: Indicates whether this is actually an array of this scalar type
        self.arr_id = arr_id

        #: Underlying C AST type declaration, used for C generation.
        self.type_decl = TypeDecl(declname=self.ident,
                                  quals=self.quals,
                                  type=self.base_type.mtype)

        #: Optional pointer declaration, used for scalar pointers.
        self.ptr_decl = PtrDecl(quals=[], type=self.type_decl)

        #: Optional array declaration, used for scalar arrays
        self.arr_decl = None
        if self.arr_id is not None:
            # Array of scalars or array of pointers (first arg is base type)
            self.arr_decl = ArrayDecl(self.ptr_decl if ptr else self.type_decl,
                                      dim=ID(name=self.arr_id),
                                      dim_quals=[])

        # TODO: Process init in some way
        #: Initial value of this scalar at the C level.
        self.init = init
        if init is not None:
            warn("Setting init is deprecated", DeprecationWarning)

        # Override superclass and re-initialize internal top-level declaration
        # Order of choosing declaration:
        # 1. If it's an array declaration (aka arr_id isn't None), use that
        # 2. Otherwise:
        #    a) If ptr is true, use the ptr declaration
        #    b) Otherwise, use the regular type declaration
        if self.arr_id is None:
            self.decl = Decl(name=self.ident,
                             quals=self.quals,
                             storage=self.storage,
                             funcspec=[],
                             type=self.ptr_decl if ptr else self.type_decl,
                             init=None,
                             bitsize=None)
        else:
            self.decl = Decl(name=self.ident,
                             quals=self.quals,
                             storage=self.storage,
                             funcspec=[],
                             type=self.arr_decl,
                             init=None,
                             bitsize=None)

    def typematch(self, other) -> bool:
        """ Checks if other is of an equivalent C type. """

        if other is None or not issubclass(other, _MyriadBase):
            return False

        if issubclass(other, MyriadScalar):
            return other.base_type is self.base_type
        else:  # TODO: Figure out what to do when given non-scalar type
            pass


class MyriadStructType(_MyriadBase):
    """
    Struct type factory.

    This class operates as a 'struct' type factory, where users declare a
    new struct type with ordered members since memory layout matters.

    Members are restricted to Myriad base types, including even other
    struct types.

    Once a `MyriadStructType` object is instantiated, it can be called as a
    function/pseudo-constructor to generate struct declarations of the
    new base C type.

    New base C types are 'registered' in the Myriad system by simply
    programmatically creating an instance class that inherits MyriadCType,
    ensuring run-time type safety checks correctly identify struct declarations
    generated from the factory instance object.
    """

    @enforce_annotations
    def __init__(self,
                 struct_name: str,
                 members: OrderedDict=None,
                 storage: list=None):
        """
        Initializes a new struct type.

        :param str struct_name: Type name of the struct.
        :param OrderedDict members: _MyriadBase-derived members of the struct
        :param list storage: List of C AST storage qualifiers (e.g. "const")
        :raises AssertionError: if members are not _MyriadBase subclassed
        """

        # Make sure we got the right parameter types
        assert_list_type(list(members.values()), _MyriadBase)

        #: Struct type identifier, i.e. the name after 'struct' in C.
        self.struct_name = struct_name

        #: Ordered members of the struct, derived from _MyriadBase.
        self.members = OrderedDict()

        members = OrderedDict() if members is None else members

        # Store member type information for introspection. Non-ordered.
        self.member_type_info = {}
        for scalar in members.values():
            self.member_type_info[scalar.ident] = scalar

        # Set struct members using ordered dict: order matters for memory!
        sorted_members = [members[idx].decl for idx in members.keys()]
        members = {v.ident: v.decl for v in members.values()}
        for member_ident, member in members.items():
            self.members[member_ident] = member

        #: pycparser C AST struct node.
        self.struct_c_ast = Struct(self.struct_name, sorted_members)

        #: Programmatically-generated base type for run-time checking.
        self.base_type = type(self.struct_name,
                              (MyriadCType,),
                              {'mtype': Struct(self.struct_name, None)})()

        # Manually build declaration to pass to super constructor
        _tmp_decl = Decl(name=None,
                         quals=[],
                         storage=[],
                         funcspec=[],
                         type=self.struct_c_ast,
                         init=None,
                         bitsize=None)
        super().__init__(None, _tmp_decl, storage=storage)

    def __call__(self,
                 ident: str,
                 ptr: bool=False,
                 quals: list=None,
                 storage: list=None,
                 init=None,
                 **kwargs) -> MyriadScalar:
        """
        Factory method for making struct instances from template.

        Further keyword arguments provided in the form of kwargs are used to
        set the initial values (if any) of the struct for static initialization
        purposes.

        :param str ident: Name of the struct variable to be created.
        :param bool ptr: Indicates whether this is a pointer to a struct
        :param list quals: C AST scope qualifiers (e.g. "static")
        :param list storage: C AST storage qualifiers (e.g. "const")
        :param init: Initial value given to this struct (TODO: NOT IMPLEMENTED)

        :return: New scalar instance of the struct type this object represents
        :rtype: MyriadScalar

        :raises TypeError: if given members' types mismatch expected types
        """

        # Assert we receive at least a subset of valid struct members to init
        prot_idents = set(self.members.keys())
        given_idents = set(kwargs.keys())
        if not given_idents.issubset(prot_idents):
            diff = given_idents - prot_idents
            msg = "{0} not valid member(s) of struct {1}"
            raise NameError(msg.format(str(diff), self.ident))

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
        # Initialize members as object attributes
        for member_name, member_val in new_kwargs.items():
            new_instance.__dict__[member_name] = member_val

        # Set struct type information for introspection
        new_instance.__dict__["struct_type_info"] = self

        return new_instance


class MyriadFunction(_MyriadBase):
    """
    Function container for Myriad functions.

    Currently MyriadFunction acts as a shell for a C AST function declaration
    node and a string body for the definition.
    """

    @enforce_annotations
    def __init__(self,
                 ident: str,
                 args_list: OrderedDict=None,
                 ret_var: MyriadScalar=None,
                 storage: list=None,
                 fun_def=None):
        """
        Initializes a new MyriadFunction.

        :param str ident: Name of the function to be created.

        :param args_list: Ordered dict of identifiers:_MyriadBase fxn args
        :type args_list: OrderedDict or None

        :param ret_var: Return value as a MyriadScalar
        :type ret_var: MyriadScalar or None

        :param storage: C AST storage qualifiers (e.g. "const")
        :type storage: list or None

        :param fun_def: Function definition in form of a string or template

        :raise AssertionError: if args_list has non-_MyriadBase members
        """

        # Always call super first
        super().__init__(ident)

        # --------------------------------------------
        # Set return value; none is given, assume void
        # --------------------------------------------
        if ret_var is None:
            ret_var = MyriadScalar(self.ident, MVoid)

        #: Return value for the function as a MyriadScalar, default MVoid.
        self.ret_var = ret_var

        # --------------------------------------------
        #  Create internal representation of args list
        # --------------------------------------------

        # Make sure we got the right parameter types
        if args_list is not None and len(args_list) > 0:
            assert_list_type(list(args_list.values()), _MyriadBase)

        #: Stores the MyriadScalars as ordered parameters.
        self.args_list = OrderedDict() if args_list is None else args_list

        #: Stores the scalar's declarations in a C AST object.
        self.param_list = ParamList([v.decl for v in self.args_list.values()])

        # ------------------------------------------
        # Create internal c_ast function declaration
        # ------------------------------------------
        _tmp_decl = copy.deepcopy(self.ret_var.decl.type)

        # Make sure we override the identifier in our copy
        if isinstance(_tmp_decl, PtrDecl):
            _tmp_decl.type.declname = self.ident
        else:
            _tmp_decl.declname = self.ident

        #: C AST function declaration with parameter list and return type.
        self.func_decl = FuncDecl(self.param_list, _tmp_decl)

        self.decl = Decl(name=self.ident,
                         quals=[],
                         storage=storage,
                         funcspec=[],
                         type=self.func_decl,
                         init=None,
                         bitsize=None)

        #: C AST typedef, must be generated using gen_typedef.
        self.fun_typedef = None
        #: Underlying MyriadCType for this function (based on typedef).
        self.base_type = None

        # Automaticall generate typedef depending on function type
        self.gen_typedef()

        #: Function definition represented as a string.
        self.fun_def = fun_def

    def copy_init(self,
                  ident: str=None,
                  args_list: OrderedDict=None,
                  ret_var: MyriadScalar=None,
                  storage: list=None,
                  fun_def=None):
        """
        Generates a "copy" of this object with modified initial values.

        :param args_list: Ordered dict of identifiers:_MyriadBase fxn args
        :type args_list: OrderedDict or None

        :param ret_var: Return value as a MyriadScalar
        :type ret_var: MyriadScalar or None

        :param storage: C AST storage qualifiers (e.g. "const")
        :type storage: list or None

        :param fun_def: Function definition in form of a string or template

        :return: Shallow copy of this object with modified initial values.
        :rtype: MyriadFunction
        """
        if ident is None:
            ident = self.ident
        if args_list is None:
            args_list = self.args_list
        if ret_var is None:
            ret_var = self.ret_var
        if storage is None:
            storage = self.storage
        if fun_def is None:
            fun_def = self.fun_def
        return MyriadFunction(ident, args_list, ret_var, storage, fun_def)

    @classmethod
    def from_method_signature(cls,
                              ident: str,
                              sig: inspect.Signature,
                              fun_body: str=None):
        """
        Converts a function signature and body into a MyriadFunction
        """
        # Convert parameter annotations into parameters themselves
        fxn_param_odict = sig.parameters.copy()
        for param_name, param in fxn_param_odict.items():
            fxn_param_odict[param_name] = param.annotation
        # Return value is the annotation; if empty, then void
        ret_var = sig.return_annotation
        if ret_var is inspect.Signature.empty:
            ret_var = None
        return cls(ident, fxn_param_odict, ret_var, fun_def=fun_body)

    def gen_typedef(self, typedef_name: str=None):
        """
        Generates an internal typedef definition for this function.

        :param typedef_name: Overrides automatic type name generation w/ value.
        """
        # Use the convention of appending _t to a type name (e.g. int64_t)
        if typedef_name is None:
            typedef_name = self.ident + "_t"

        _tmp, tmp = None, None

        # Fix for an insiduous bug with string identifiers (e.g. int64_t)
        if isinstance(self.func_decl.type.type, TypeDecl):
            _tmp = self.func_decl.type.type.type
        else:
            _tmp = IdentifierType(names=self.func_decl.type.type.names)

        # Use TypeDecl/PtrDecl depending on whether return value is a pointer
        if self.ret_var.ptr:
            tmp = PtrDecl([], TypeDecl(typedef_name, [], _tmp))
        else:
            tmp = TypeDecl(typedef_name, [], _tmp)

        _tmp_fdecl = PtrDecl([], FuncDecl(self.param_list, tmp))

        # Update base type so its registered a MyriadCType subclass
        self.base_type = type(typedef_name,
                              (MyriadCType,),
                              {'mtype': IdentifierType([typedef_name])})()

        # Create typedef so its registered by the pycparser AST
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
        if isinstance(self.fun_def, str):
            return self.fun_def
        else:
            raise NotImplementedError("Non-string representations unsupported")


def cast_to_parent(struct: MyriadStructType,
                   field: str) -> str:
    """
    Casts a type to a parent type to allow for accessing a given field.
    Raises an exception if no possible cast exists to allow this.

    :param struct: Struct type in which to look for field
    :type struct: MyriadStructType

    :param field: Field which we are looking for.
    :type field: str

    :return: String representing the cast, e.g. (double*) or (struct sc*)
    :rtype: str
    """
    if field in struct.members:
        return '(struct ' + struct.struct_name + '*)'
    # i.e. if we have a member named '_', and it has a struct type
    elif ('_' in struct.members and
          "struct_type_info" in struct.member_type_info['_'].__dict__):
        return cast_to_parent(struct.member_type_info['_'].struct_type_info,
                              field)
    else:
        raise Exception("Field not found in struct type")

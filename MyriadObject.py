#!/usr/bin/python3
"""
TODO: Docstring
"""

from collections import OrderedDict

from pycparser.c_ast import Decl, TypeDecl, Struct, PtrDecl

from myriad_module import MyriadModule, MyriadMethod
from myriad_types import MyriadScalar, MyriadFunction, MyriadStructType
from myriad_types import MVoid, MInt, MVarArgs, MSizeT


# pylint: disable=R0902
class MyriadObject(MyriadModule):
    """
    Special initialization for core object.
    """

    # TODO: Probably make these templates a little more generic...
    CTOR_TEMPLATE = """
    return _self;
    """

    DTOR_TEMPLATE = """
    free(_self);
    return EXIT_SUCCESS;
    """

    CUDAFY_TEMPLATE = """
    #ifdef CUDA
    {
        struct MyriadObject* self = (struct MyriadObject*) self_obj;
        void* n_dev_obj = NULL;
	size_t my_size = myriad_size_of(self);

	const struct MyriadClass* tmp = self->m_class;
	self->m_class = self->m_class->device_class;

	CUDA_CHECK_RETURN(cudaMalloc(&n_dev_obj, my_size));

	CUDA_CHECK_RETURN(
		cudaMemcpy(
			n_dev_obj,
			self,
			my_size,
			cudaMemcpyHostToDevice
			)
		);

	self->m_class = tmp;

	return n_dev_obj;
    }
    #else
    {
        return NULL;
    }
    #endif
    """

    DECUDAFY_TEMPLATE = """
    return;
    """

    # pylint: disable=R0914
    def __init__(self):
        # Set CUDA support status
        self.cuda = True

        # Set internal names for classes
        self.obj_name = "MyriadObject"
        self.cls_name = "MyriadClass"

        # ---------------------------------------------------------------------
        # Cheat by hardcoding methods in constructor
        # ---------------------------------------------------------------------

        # TODO: Hardcode instance methods
        self.methods = set()
        _self = MyriadScalar("self", MVoid, True, quals=["const"])

        # extern void* myriad_ctor(void* _self, va_list* app);
        _app = MyriadScalar("app", MVarArgs, ptr=True)
        _ret_var = MyriadScalar('', MVoid, ptr=True)
        _ctor_fun = MyriadFunction("myriad_ctor",
                                   OrderedDict({0: _self, 1: _app}),
                                   ret_var=_ret_var)
        self.methods.add(MyriadMethod(_ctor_fun,
                                      MyriadObject.CTOR_TEMPLATE,
                                      self.obj_name))

        # extern int myriad_dtor(void* _self);
        _ret_var = MyriadScalar('', MInt)
        _dtor_fun = MyriadFunction("myriad_dtor",
                                   OrderedDict({0: _self}),
                                   ret_var=_ret_var)
        self.methods.add(MyriadMethod(_dtor_fun,
                                      MyriadObject.DTOR_TEMPLATE,
                                      self.obj_name))

        # extern void* myriad_cudafy(void* _self, int clobber);
        _clobber = MyriadScalar("clobber", MInt)
        _ret_var = MyriadScalar('', MVoid, ptr=True)
        _cudafy_fun = MyriadFunction("myriad_cudafy",
                                     OrderedDict({0: _self, 1: _clobber}),
                                     ret_var=_ret_var)
        self.methods.add(MyriadMethod(_cudafy_fun,
                                      MyriadObject.CUDAFY_TEMPLATE,
                                      self.obj_name))

        # extern void myriad_decudafy(void* _self, void* cu_self);
        _cu_self = MyriadScalar("cu_self", MVoid, ptr=True)
        _decudafy_fun = MyriadFunction("myriad_decudafy",
                                       OrderedDict({0: _self, 1: _cu_self}))
        self.methods.add(MyriadMethod(_decudafy_fun,
                                      MyriadObject.DECUDAFY_TEMPLATE,
                                      self.obj_name))

        # ---------------------------------------------------------------------
        # Initialize class object and object class
        # ---------------------------------------------------------------------

        # Cheat here by hand-crafting our own object/class variables
        def _gen_mclass_ptr_scalar(ident: str):
            """ Quick-n-Dirty way of hard-coding MyriadClass struct ptrs. """
            tmp = MyriadScalar(ident,
                               MVoid,
                               True,
                               quals=["const"])
            tmp.type_decl = TypeDecl(declname=ident,
                                     quals=[],
                                     type=Struct("MyriadClass", None))
            tmp.ptr_decl = PtrDecl(quals=[],
                                   type=tmp.type_decl)
            tmp.decl = Decl(name=ident,
                            quals=["const"],
                            storage=[],
                            funcspec=[],
                            type=tmp.ptr_decl,
                            init=None,
                            bitsize=None)
            return tmp

        obj_vars = OrderedDict({0: _gen_mclass_ptr_scalar("m_class")})

        self.obj_struct = MyriadStructType(self.obj_name, obj_vars)

        # Initialize class variables, i.e. function pointers for methods
        cls_vars = OrderedDict()
        cls_vars[0] = self.obj_struct("_", quals=["const"])
        cls_vars[1] = _gen_mclass_ptr_scalar("super")
        cls_vars[2] = _gen_mclass_ptr_scalar("device_class")
        cls_vars[3] = MyriadScalar("size", MSizeT)

        for indx, method in enumerate(self.methods):
            m_scal = MyriadScalar("my_" + method.delegator.fun_typedef.name,
                                  method.delegator.base_type)
            cls_vars[indx+4] = m_scal

        self.cls_vars = cls_vars
        self.cls_struct = MyriadStructType(self.cls_name, self.cls_vars)

        # --------------------------------------------------------------------

        self.functions = set()
        self._init_module_funs()

        # --------------------------------------------------------------------

        # Initialize module global variables
        self.module_vars = set()
        v_obj = MyriadScalar(self.obj_name,
                             MVoid,
                             True,
                             quals=["const"])
        self.module_vars.add(v_obj)
        v_cls = MyriadScalar(self.cls_name,
                             MVoid,
                             True,
                             quals=["const"])
        self.module_vars.add(v_cls)

        # Initialize standard library imports, by default with fail-safes
        self.lib_includes = MyriadModule.DEFAULT_LIB_INCLUDES

        # TODO: Initialize local header imports
        self.local_includes = set()

        # Initialize C header template
        self.header_template = None
        self.initialize_header_template()

    def _init_module_funs(self):
        _init_cu = MyriadFunction("initCUDAObjects",
                                  None,
                                  MyriadScalar('', MInt),
                                  fun_def=None)
        self.functions.add(_init_cu)
        """
extern int initCUDAObjects();

extern void* myriad_new(const void* _class, ...);

extern const void* myriad_class_of(const void* _self);

extern size_t myriad_size_of(const void* self);

extern int myriad_is_a(const void* _self, const struct MyriadClass* m_class);

extern int myriad_is_of(const void* _self, const struct MyriadClass* m_class);

extern const void* myriad_super(const void* _self);
        """
        pass

def create_myriad_object():
    obj = MyriadObject()
    obj.render_header_template(True)


def main():
    create_myriad_object()


if __name__ == "__main__":
    main()

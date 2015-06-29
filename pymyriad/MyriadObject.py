#!/usr/bin/python3
"""
TODO: Docstring
"""

from copy import deepcopy
from collections import OrderedDict

from pycparser.c_ast import Decl, TypeDecl, Struct, PtrDecl, ParamList
from pycparser.c_ast import EllipsisParam, FuncDecl

import myriad_module
from myriad_module import MyriadModule, MyriadMethod
from myriad_types import MyriadScalar, MyriadFunction, MyriadStructType
from myriad_types import MVoid, MInt, MVarArgs, MSizeT


# pylint: disable=R0902,R0903
class _MyriadObject(MyriadModule):
    """
    Special initialization for core object.
    """

    # TODO: Probably make these templates a little more generic...
    OBJ_CTOR_T = """
    return self;
    """

    CLS_CTOR_T = """
    struct MyriadClass* _self = (struct MyriadClass*) self;
    const size_t offset = offsetof(struct MyriadClass, my_myriad_ctor_t);

    _self->super = va_arg(*app, struct MyriadClass*);
    _self->size = va_arg(*app, size_t);

    assert(_self->super);

    memcpy((char*) _self + offset,
           (char*) _self->super + offset,
           myriad_size_of(_self->super) - offset);

    va_list ap;
    va_copy(ap, *app);
    typedef void (* voidf) ();
    voidf selector = NULL; selector = va_arg(ap, voidf);

    while (selector)
    {
        const voidf curr_method = va_arg(ap, voidf);

        if (selector == (voidf) myriad_ctor)
        {
            *(voidf *) &_self->my_myriad_ctor_t = curr_method;
        } else if (selector == (voidf) myriad_cudafy) {
            *(voidf *) &_self->my_myriad_cudafy_t = curr_method;
        } else if (selector == (voidf) myriad_dtor) {
            *(voidf *) &_self->my_myriad_dtor_t = curr_method;
        } else if (selector == (voidf) myriad_decudafy) {
            *(voidf *) &_self->my_myriad_decudafy_t = curr_method;
        }

        selector = va_arg(ap, voidf);
    }

    return _self;
    """

    OBJ_DTOR_T = """
    free(self);
    return EXIT_SUCCESS;
    """

    CLS_DTOR_T = """
    fprintf(stderr, "Destroying a Class is undefined behavior.");
    return EXIT_FAILURE;
    """

    OBJ_CUDAFY_T = """
    #ifdef CUDA
    {
        struct MyriadObject* _self = (struct MyriadObject*) self;
        void* n_dev_obj = NULL;
        size_t my_size = myriad_size_of(_self);

        const struct MyriadClass* tmp = _self->m_class;
        _self->m_class = _self->m_class->device_class;

        CUDA_CHECK_RETURN(cudaMalloc(&n_dev_obj, my_size));

        CUDA_CHECK_RETURN(
                cudaMemcpy(
                        n_dev_obj,
                        _self,
                        my_size,
                        cudaMemcpyHostToDevice
                        )
                );

        _self->m_class = tmp;

        return n_dev_obj;
    }
    #else
    {
        return NULL;
    }
    #endif
    """

    CLS_CUDAFY_T = """
    #ifdef CUDA
    {
        struct MyriadClass* _self = (struct MyriadClass*) self;

        const struct MyriadClass* dev_class = NULL;

        const size_t class_size = myriad_size_of(_self);

        CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_class, class_size));

        const struct MyriadClass* class_cpy = (const struct MyriadClass*) calloc(1, class_size);
        memcpy((void*)class_cpy, self, class_size);

        memcpy((void*)&class_cpy->_.m_class, &dev_class, sizeof(void*));

        CUDA_CHECK_RETURN(
            cudaMemcpy(
                (void*)dev_class,
                class_cpy,
                class_size,
                cudaMemcpyHostToDevice
                )
            );

        free((void*)class_cpy);

        return (void*) dev_class;
    }
    #else
    {
        return NULL;
    }
    #endif
    """

    OBJ_DECUDAFY_T = """
    return;
    """

    CLS_DECUDAFY_T = """
    fprintf(stderr, "De-CUDAfying a class is undefined behavior. Aborted.");
    return;
    """

    # -------------------------------------------------------------------------
    # Global function templates
    # -------------------------------------------------------------------------

    MYRIAD_NEW_T = """
    const struct MyriadClass* prototype_class = (const struct MyriadClass*) _class;
    struct MyriadObject* curr_obj;
    va_list ap;

    assert(prototype_class && prototype_class->size);

    curr_obj = (struct MyriadObject*) _my_calloc(1, prototype_class->size);
    assert(curr_obj);

    curr_obj->m_class = prototype_class;

    va_start(ap, _class);
    curr_obj = (struct MyriadObject*) myriad_ctor(curr_obj, &ap);
    va_end(ap);

    return curr_obj;
    """

    MYRIAD_CLASS_OF_T = """
    const struct MyriadObject* _self = (const struct MyriadObject*) self;
    return _self->m_class;
    """

    MYRIAD_SIZE_OF_T = """
    const struct MyriadClass* m_class = (const struct MyriadClass*) myriad_class_of(self);
    return m_class->size;
    """

    MYRIAD_IS_A_T = """
    return self && myriad_class_of(self) == m_class;
    """

    MYRIAD_IS_OF_T = """
    if (self)
    {   
        const struct MyriadClass * myClass = (const struct MyriadClass*) myriad_class_of(self);

        if (m_class != MyriadObject)
        {
            while (myClass != m_class)
            {
                if (myClass != MyriadObject)
                {
                    myClass = (const struct MyriadClass*) myriad_super(myClass);
                } else {
                    return 0;
                }
            }
        }

        return 1;
    }

    return 0;
    """

    MYRIAD_INIT_CUDA_T = """
    #ifdef CUDA
    {
        const struct MyriadClass *obj_addr = NULL, *class_addr = NULL;
        const size_t obj_size = sizeof(struct MyriadObject);
        const size_t class_size = sizeof(struct MyriadClass);

        CUDA_CHECK_RETURN(cudaMalloc((void**)&obj_addr, class_size));
        CUDA_CHECK_RETURN(cudaMalloc((void**)&class_addr, class_size));

        const struct MyriadClass anon_class_class = {
            {class_addr},
            obj_addr,
            class_addr,
            class_size,
            NULL,
            NULL,
            NULL,
            NULL,
        };

        CUDA_CHECK_RETURN(
            cudaMemcpy(
                (void**) class_addr,
                &anon_class_class,
                sizeof(struct MyriadClass),
                cudaMemcpyHostToDevice
                )
            );

        object[1].device_class = class_addr;

        const struct MyriadClass anon_obj_class = {
            {class_addr},
            obj_addr,
            class_addr,
            obj_size,
            NULL,
            NULL,
            NULL,
            NULL,
        };

        CUDA_CHECK_RETURN(
            cudaMemcpy(
                (void**) obj_addr,
                &anon_obj_class,
                sizeof(struct MyriadClass),
                cudaMemcpyHostToDevice
                )
            );

        object[0].device_class = (const struct MyriadClass*) obj_addr;

        CUDA_CHECK_RETURN(
            cudaMemcpyToSymbol(
                (const void*) &MyriadClass_dev_t,
                &class_addr,
                sizeof(void*),
                0,
                cudaMemcpyHostToDevice
                )
            );

        CUDA_CHECK_RETURN(
            cudaMemcpyToSymbol(
                (const void*) &MyriadObject_dev_t,
                &obj_addr,
                sizeof(void*),
                0,
                cudaMemcpyHostToDevice
                )
            );

        return 0;
    }
    #else
    {
        return EXIT_FAILURE;
    }
    #endif
    """

    MYRIAD_SUPER_T = """
    const struct MyriadClass* _self = (const struct MyriadClass*) self;

    assert(_self && _self->super);
    return _self->super;
    """

    @staticmethod
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

    def __init__(self):
        # Set CUDA support status
        self.cuda = True

        # Set internal names for classes
        self.obj_name = "MyriadObject"
        self.cls_name = "MyriadClass"

        # Cheat by hardcoding methods in constructor
        self.methods = OrderedDict()
        self._setup_methods()

        # Hardcode functions, too, while we're at it
        self.functions = OrderedDict()
        self._init_module_funs()

        # ---------------------------------------------------------------------
        # Initialize class object and object class
        # ---------------------------------------------------------------------

        obj_vars = {0: _MyriadObject._gen_mclass_ptr_scalar("m_class")}
        obj_vars = OrderedDict(obj_vars)

        self.obj_struct = MyriadStructType(self.obj_name, obj_vars)

        # Initialize class variables, i.e. function pointers for methods
        cls_vars = OrderedDict()
        cls_vars[0] = self.obj_struct("_", quals=["const"])
        cls_vars[1] = _MyriadObject._gen_mclass_ptr_scalar("super")
        cls_vars[2] = _MyriadObject._gen_mclass_ptr_scalar("device_class")
        cls_vars[3] = MyriadScalar("size", MSizeT)

        for indx, method in enumerate(self.methods.values()):
            m_scal = MyriadScalar("my_" + method.delegator.fun_typedef.name,
                                  method.delegator.base_type)
            cls_vars[indx+4] = m_scal

        self.cls_struct = MyriadStructType(self.cls_name, cls_vars)

        # --------------------------------------------------------------------

        # Initialize module global variables

        self.module_vars = OrderedDict()
        self._init_module_vars()

        # Initialize standard library imports, by default with fail-safes
        self.lib_includes = MyriadModule.DEFAULT_LIB_INCLUDES

        # TODO: Initialize local header imports
        self.local_includes = set()

        # Initialize C header template
        self.header_template = \
            self.create_file_template(".h", myriad_module.HEADER_FILE_TEMPLATE)

        self.cuda_header_template = \
            self.create_file_template(".cuh", myriad_module.CUH_FILE_TEMPLATE)

        self.c_file_template = \
            self.create_file_template(".c", myriad_module.C_FILE_TEMPLATE)

    def _setup_methods(self):
        """ Hardcode of the various methods w/ pre-written templates. """

        # Everyone uses self...
        _self = MyriadScalar("self", MVoid, True, quals=["const"])

        # extern void* myriad_ctor(void* _self, va_list* app);
        _app = MyriadScalar("app", MVarArgs, ptr=True)
        _ret_var = MyriadScalar('', MVoid, ptr=True)
        _ctor_fun = MyriadFunction("myriad_ctor",
                                   OrderedDict({0: _self, 1: _app}),
                                   ret_var=_ret_var)
        _dict = {self.obj_name: _MyriadObject.OBJ_CTOR_T,
                 self.cls_name: _MyriadObject.CLS_CTOR_T}
        self.methods["myriad_ctor"] = MyriadMethod(_ctor_fun, _dict)

        # extern int myriad_dtor(void* _self);
        _ret_var = MyriadScalar('', MInt)
        _dtor_fun = MyriadFunction("myriad_dtor",
                                   OrderedDict({0: _self}),
                                   ret_var=_ret_var)
        _dict = {self.obj_name: _MyriadObject.OBJ_DTOR_T,
                 self.cls_name: _MyriadObject.CLS_DTOR_T}
        self.methods["myriad_dtor"] = MyriadMethod(_dtor_fun, _dict)

        # extern void* myriad_cudafy(void* _self, int clobber);
        _clobber = MyriadScalar("clobber", MInt)
        _ret_var = MyriadScalar('', MVoid, ptr=True)
        _cudafy_fun = MyriadFunction("myriad_cudafy",
                                     OrderedDict({0: _self, 1: _clobber}),
                                     ret_var=_ret_var)
        _dict = {self.obj_name: _MyriadObject.OBJ_CUDAFY_T,
                 self.cls_name: _MyriadObject.CLS_CUDAFY_T}
        self.methods["myriad_cudafy"] = MyriadMethod(_cudafy_fun, _dict)

        # extern void myriad_decudafy(void* _self, void* cu_self);
        _cu_self = MyriadScalar("cu_self", MVoid, ptr=True)
        _decudafy_fun = MyriadFunction("myriad_decudafy",
                                       OrderedDict({0: _self, 1: _cu_self}))
        _dict = {self.obj_name: _MyriadObject.OBJ_DECUDAFY_T,
                 self.cls_name: _MyriadObject.CLS_DECUDAFY_T}
        self.methods["myriad_decudafy"] = MyriadMethod(_decudafy_fun, _dict)

    def _init_module_vars(self):
        SPEC_TEMPLATE = """
static struct MyriadClass object[] =
{
    {
        { object + 1 },
        object,
        NULL,
        sizeof(struct MyriadObject),
        MyriadObject_myriad_ctor,
        MyriadObject_myriad_dtor,
        MyriadObject_myriad_cudafy,
        MyriadObject_myriad_decudafy,
    },
    {
        { object + 1 },
        object,
        NULL,
        sizeof(struct MyriadClass),
        MyriadClass_myriad_ctor,
        MyriadClass_myriad_dtor,
        MyriadClass_myriad_cudafy,
        MyriadClass_myriad_decudafy,
    }
};
        """
        self.module_vars['object'] = SPEC_TEMPLATE
        v_obj = MyriadScalar(self.obj_name,
                             MVoid,
                             True,
                             quals=["const"],
                             init="object")
        self.module_vars['MyriadObject'] = v_obj
        v_cls = MyriadScalar(self.cls_name,
                             MVoid,
                             True,
                             quals=["const"],
                             init="object + 1")
        self.module_vars['MyriadClass'] = v_cls

    def _init_module_funs(self):
        """ Hardcode module functions using pre-made templates. """

        # Some functions share these; best we save the heap space
        _m_class = _MyriadObject._gen_mclass_ptr_scalar("m_class")
        _self = MyriadScalar("self", MVoid, True, quals=["const"])

        # int initCUDAObjects()
        _ret_var = MyriadScalar('', MInt)
        self.functions["initCUDAObjects"] = MyriadFunction("initCUDAObjects",
                                                           None,
                                                           _ret_var,
                                                           None,
                                                           _MyriadObject.MYRIAD_INIT_CUDA_T)

        # const void* myriad_class_of(const void* _self)
        _ret_var = MyriadScalar('', MVoid, True)
        self.functions["myriad_class_of"] = MyriadFunction("myriad_class_of",
                                                           OrderedDict({0: _self}),
                                                           _ret_var,
                                                           None,
                                                           _MyriadObject.MYRIAD_CLASS_OF_T)

        # size_t myriad_size_of(const void* self);
        _ret_var = MyriadScalar('', MSizeT)
        self.functions["myriad_size_of"] = MyriadFunction("myriad_size_of",
                                                          OrderedDict({0: _self}),
                                                          _ret_var,
                                                          None,
                                                          _MyriadObject.MYRIAD_SIZE_OF_T)

        # int myriad_is_a(const void* _self, const struct MyriadClass* m_class)
        _ret_var = MyriadScalar('', MInt)
        _args = OrderedDict({0: _self, 1: _m_class})
        self.functions["myriad_is_a"] = MyriadFunction("myriad_is_a",
                                                       _args,
                                                       _ret_var,
                                                       None,
                                                       _MyriadObject.MYRIAD_IS_A_T)

        # int myriad_is_of(const void* _self,const struct MyriadClass* m_class)
        _ret_var = MyriadScalar('', MInt)
        _args = OrderedDict({0: _self, 1: _m_class})
        self.functions["myriad_is_of"] = MyriadFunction("myriad_is_of",
                                                        _args,
                                                        _ret_var,
                                                        None,
                                                        _MyriadObject.MYRIAD_IS_OF_T)

        # extern const void* myriad_super(const void* _self);
        _ret_var = MyriadScalar('', MVoid, ptr=True, quals=['const'])
        self.functions["myriad_super"] = MyriadFunction("myriad_super",
                                                        OrderedDict({0: _self}),
                                                        _ret_var,
                                                        None,
                                                        _MyriadObject.MYRIAD_SUPER_T)

        # extern void* myriad_new(const void* _class, ...);
        # TODO: Make sure myriad_new works
        _ret_var = MyriadScalar('', MVoid, ptr=True)
        _vclass = MyriadScalar("_class", MVoid, ptr=True, quals=["const"])
        _new = MyriadFunction("myriad_new",
                              OrderedDict(),
                              _ret_var,
                              fun_def=_MyriadObject.MYRIAD_NEW_T)
        _new.param_list = ParamList([_vclass.decl, EllipsisParam()])
        _tmp_decl = deepcopy(_new.ret_var.decl.type)
        _tmp_decl.type.declname = _new.ident
        _new.func_decl = FuncDecl(_new.param_list, _tmp_decl)
        _new.decl = Decl(name=_new.ident,
                         quals=[],
                         storage=[],
                         funcspec=[],
                         type=_new.func_decl,
                         init=None,
                         bitsize=None)
        self.functions["myriad_new"] = _new


class MyriadObject(object):
    """
    Dummy class used to obfuscate the ugly internals of the module system
    from the end user, as well as providing an easy superclass target.
    """

    _my_module = _MyriadObject()


def main():
    # TESTING
    MyriadObject._my_module.header_template.render_to_file()
    MyriadObject._my_module.c_file_template.render_to_file()
    MyriadObject._my_module.cuda_header_template.render_to_file()

if __name__ == "__main__":
    main()

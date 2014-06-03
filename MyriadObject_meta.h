#ifndef MYRIADOBJECT_META_H
#define MYRIADOBJECT_META_H

#include "myriad_metaprogramming.h"

// Generics
#define MYRIADOBJECT_OBJECT MyriadObject
#define MYRIADOBJECT_CLASS MyriadClass

// Attributes
#define OBJECTS_CLASS m_class
#define EMBEDDED_OBJECT_NAME _
#define SUPERCLASS super
#define ONDEVICE_CLASS device_class
#define OBJECTS_SIZE size
#define CONSTRUCTOR my_ctor
#define DESTRUCTOR my_dtor
#define CUDAFIER my_cudafy
#define DECUDAFIER my_decudafy

#define DEV_T dev_t

// Ctor fun properties
#define CTOR_FUN_NAME ctor
#define CTOR_FUN_TYPEDEF_NAME _MYRIAD_CAT(CTOR_FUN_NAME,_t)
#define CTOR_FUN_ARGS void* self, va_list* app
#define CTOR_FUN_RET void*

// Dtor fun properties
#define DTOR_FUN_NAME dtor
#define DTOR_FUN_TYPEDEF_NAME _MYRIAD_CAT(DTOR_FUN_NAME,_t)
#define DTOR_FUN_ARGS void* self
#define DTOR_FUN_RET int

// Cudafy fun properties
#define CUDAFY_FUN_NAME cudafy
#define CUDAFY_FUN_TYPEDEF_NAME _MYRIAD_CAT(CUDAFY_FUN_NAME,_t)
//#define CUDAFY_FUN_ARGS void* self_obj, int clobber ORIGINALLY THIS
#define CUDAFY_FUN_ARGS void* _self, int clobber
#define CUDAFY_FUN_RET void*

// Decudafy fun properties
#define DECUDAFY_FUN_NAME decudafy
#define DECUDAFY_FUN_TYPEDEF_NAME _MYRIAD_CAT(DECUDAFY_FUN_NAME,_t)
#define DECUDAFY_FUN_ARGS void* _self, void* cuda_self
#define DECUDAFY_FUN_RET void

// Super fun properties
#define SUPER_FUN_NAME SUPERCLASS
#define SUPER_FUN_RET const void*
#define SUPER_FUN_ARGS const void* _self

// SuperCtor fun properties
#define SUPERCTOR_FUN_NAME ctor
#define SUPERCTOR_FUN_RET void*
#define SUPERCTOR_FUN_ARGS const void* _class, void* _self, va_list* app
    
// SuperDtor fun properties
#define SUPERDTOR_FUN_NAME dtor
#define SUPERDTOR_FUN_RET int
#define SUPERDTOR_FUN_ARGS const void* _class, void* _self
    
// SuperCudafy fun properties
#define SUPERCUDAFY_FUN_NAME cudafy
#define SUPERCUDAFY_FUN_RET void*
#define SUPERCUDAFY_FUN_ARGS const void* _class, void* _self, int clobber
    
// SuperDecudafy fun properties
#define SUPERDECUDAFY_FUN_NAME decudafy
#define SUPERDECUDAFY_FUN_RET void
#define SUPERDECUDAFY_FUN_ARGS const void* _class, void* _self, void* cuda_self
    
#define SUPERCLASS_CTOR MYRIAD_CAT(SUPERCLASS, MYRIAD_CAT(_, SUPERCTOR_FUN_NAME))
#define SUPERCLASS_DTOR MYRIAD_CAT(SUPERCLASS, MYRIAD_CAT(_, SUPERDTOR_FUN_NAME))
#define SUPERCLASS_CUDAFY MYRIAD_CAT(SUPERCLASS, MYRIAD_CAT(_, SUPERCUDAFY_FUN_NAME))
#define SUPERCLASS_DECUDAFY MYRIAD_CAT(SUPERCLASS, MYRIAD_CAT(_, SUPERDECUDAFY_FUN_NAME))

// Dynamic initialisation properties
#define DYNAMIC_INIT_FXN_RET void
#define DYNAMIC_INIT_FXN_ARGS int init_cuda

#endif

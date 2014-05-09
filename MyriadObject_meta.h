#ifndef MYRIADOBJECT_META_H
#define MYRIADOBJECT_META_H

#include "myriad_metaprogramming.h"

// Generics
#define SELF_NAME self
#define SELF_TYPE void*
#define _SELF_NAME _MYRIAD_CAT(_,SELF_NAME)

// Generic fun properties
#define GENERIC_FUN_NAME voidf
#define GENERIC_FUN_ARGS 
#define GENERIC_FUN_RET void

// Ctor fun properties
#define CTOR_FUN_NAME ctor
#define CTOR_FUN_TYPEDEF_NAME _MYRIAD_CAT(CTOR_FUN_NAME,_t)
#define CTOR_FUN_ARGS SELF_TYPE SELF_NAME, va_list* app
#define CTOR_FUN_RET void*

// Dtor fun properties
#define DTOR_FUN_NAME dtor
#define DTOR_FUN_TYPEDEF_NAME _MYRIAD_CAT(DTOR_FUN_NAME,_t)
#define DTOR_FUN_ARGS SELF_TYPE SELF_NAME
#define DTOR_FUN_RET int

// Cudafy fun properties
#define CUDAFY_FUN_NAME cudafy
#define CUDAFY_FUN_TYPEDEF_NAME _MYRIAD_CAT(CUDAFY_FUN_NAME,_t)
#define CUDAFY_FUN_ARGS void* self_obj, int clobber
#define CUDAFY_FUN_RET void*

// Decudafy fun properties
#define DECUDAFY_FUN_NAME decudafy
#define DECUDAFY_FUN_TYPEDEF_NAME _MYRIAD_CAT(DECUDAFY_FUN_NAME,_t)
#define DECUDAFY_FUN_ARGS void* _self, void* cuda_self
#define DECUDAFY_FUN_RET void

#endif

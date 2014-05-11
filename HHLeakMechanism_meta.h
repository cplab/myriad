#ifndef HHLEAKMECHANISM_META_H
#define HHLEAKMECHANISM_META_H

#include "myriad_metaprogramming.h"

// Generics 
#define SELF_NAME self
#define SELF_TYPE void*
#define _SELF_NAME _MYRIAD_CAT(_,SELF_NAME)
#define OBJECT_NAME HHLeakMechanism
#define OBJECT_NAME_POINTER HHLeakMechanism*

// Ctor fun properties
#define CTOR_FUN_NAME ctor
#define CTOR_FUN_TYPEDEF_NAME _MYRIAD_CAT(CTOR_FUN_NAME,_t)
#define CTOR_FUN_ARGS SELF_TYPE SELF_NAME, va_list* app
#define CTOR_FUN_RET void*

// Cudafy fun properties
#define CUDAFY_FUN_NAME cudafy
#define CUDAFY_FUN_TYPEDEF_NAME _MYRIAD_CAT(CUDAFY_FUN_NAME,_t)
#define CUDAFY_FUN_ARGS void* _SELF_NAME, int clobber
#define CUDAFY_FUN_RET void*





#endif
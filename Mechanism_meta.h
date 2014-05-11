#ifndef MECHANISM_META_H
#define MECHANISM_META_H

#include "myriad_metaprogramming.h"

// Generics
#define SELF_NAME self
#define SELF_TYPE void*
#define _SELF_NAME _MYRIAD_CAT(_,SELF_NAME)
#define OBJECT_NAME Mechanism

// Ctor fun properties
#define CTOR_FUN_NAME ctor
#define CTOR_FUN_TYPEDEF_NAME _MYRIAD_CAT(CTOR_FUN_NAME,_t)
#define CTOR_FUN_ARGS SELF_TYPE SELF_NAME, va_list* app
#define CTOR_FUN_RET void*

#endif

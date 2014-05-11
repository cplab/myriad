/**
   @author Alex J Davies
 */

#ifndef COMPARTMENT_META_H
#define COMPARTMENT_META_H

#include "myriad_metaprogramming.h"

// Compartment Ctor fun properties
#define CTOR_FUN_NAME ctor
#define CTOR_FUN_TYPEDEF_NAME _MYRIAD_CAT(CTOR_FUN_NAME,_t)
#define CTOR_FUN_ARGS SELF_TYPE SELF_NAME, va_list* app
#define CTOR_FUN_RET void*
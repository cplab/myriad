## Add include guards
<% include_guard = obj_name.upper() + "_H" %>
#ifndef ${include_guard}
#define ${include_guard}

## Top-level Myriad include
#include "myriad.h"

## Add library includes
% for lib in lib_includes:
#include <${lib}>
% endfor

## Add local includes
% for lib in local_includes:
#include "${lib}"
% endfor

## Class/Object structs
${obj_struct.stringify_decl()};

${cls_struct.stringify_decl()};


## Declare typedefs/vtables/init functions for own methods ONLY
% for (delg, _) in own_method_delgs:

#ifndef ${delg.typedef_name.upper()}
#define ${delg.typedef_name.upper()}
${delg.stringify_typedef()};
#endif

extern
#ifdef CUDA
__constant__
#endif
const ${delg.typedef_name} ${delg.ident}_vtable[NUM_CU_CLASS];

## Top-level init functions for vtable
extern void init_${delg.ident}_cuvtable(void);

% endfor

## Method delegators
% for delg, super_delg in own_method_delgs:
extern ${delg.stringify_decl()};

extern ${super_delg.stringify_decl()};

% endfor

#endif

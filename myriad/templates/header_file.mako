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

## Declare typedefs for own methods ONLY
% for (delg, _) in own_method_delgs:
#ifndef ${delg.typedef_name.upper()}
#define ${delg.typedef_name.upper()}
${delg.stringify_typedef()};
#endif
% endfor

## Struct forward declarations
struct ${cls_name};
struct ${obj_name};

## Top-level init function
extern void init${obj_name}(void);

## Top-level pointers for myriad_new purposes
extern const void* ${obj_name};
extern const void* ${cls_name};

## Method delegators
% for delg, super_delg in own_method_delgs:
extern ${delg.stringify_decl()};

extern ${super_delg.stringify_decl()};

% endfor

## Class/Object structs
${obj_struct.stringify_decl()};

${cls_struct.stringify_decl()};

#endif

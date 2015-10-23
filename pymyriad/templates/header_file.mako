<%!
    import myriad_types
    import myriad_metaclass
%>

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

## Declare typedefs
% for method in myriad_methods:
${method.delegator.stringify_typedef()};
% endfor

## Struct forward declarations
struct ${cls_name};
struct ${obj_name};

## Top-level functions
## % for fun in functions.values():
## extern ${fun.stringify_decl()};
## % endfor

## Method delegators
% for method in myriad_methods:
<%
    delg = myriad_metaclass.create_delegator(method, cls_name)
    super_delg = myriad_metaclass.create_super_delegator(delg, cls_name)
%>
extern ${delg.stringify_decl()};

extern ${super_delg.stringify_decl()};

% endfor

## Class/Object structs
${obj_struct.stringify_decl()};
${cls_struct.stringify_decl()};

#endif

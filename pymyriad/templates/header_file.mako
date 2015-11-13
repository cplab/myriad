<%!
    from myriad_metaclass import create_delegator, create_super_delegator
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

## Declare typedefs for own methods ONLY
% for delg in [create_delegator(m, cls_name) for m in own_methods]:
${delg.stringify_typedef()};
% endfor

## Struct forward declarations
struct ${cls_name};
struct ${obj_name};

## Top-level init function
extern void init${obj_name}(void);

## Method delegators
% for method in own_methods:
<%
    delg = create_delegator(method, cls_name)
    super_delg = create_super_delegator(delg, cls_name)
%>
extern ${delg.stringify_decl()};

extern ${super_delg.stringify_decl()};

% endfor

## Class/Object structs
${obj_struct.stringify_decl()};

${cls_struct.stringify_decl()};

#endif

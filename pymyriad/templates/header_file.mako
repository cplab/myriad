<%!
    import myriad_types
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
% for method in methods.values():
    % if not method.inherited:
${method.delegator.stringify_typedef()};
    % endif
% endfor

## Struct forward declarations
struct ${cls_name};
struct ${obj_name};

## Module variables
% for m_var in module_vars.values():
    % if type(m_var) is not str and 'static' not in m_var.decl.storage:
extern ${m_var.stringify_decl()};
    % endif
% endfor

## Top-level functions
% for fun in functions.values():
extern ${fun.stringify_decl()};
% endfor

## Method delegators
% for method in [m for m in methods.values() if not m.inherited]:

extern ${method.delegator.stringify_decl()};

extern ${method.super_delegator.stringify_decl()};

% endfor

## Class/Object structs
${obj_struct.stringify_decl()};
${cls_struct.stringify_decl()};

#endif

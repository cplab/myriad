## Python imports as a module-level block
<%!
    from context import myriad
    from myriad.myriad_metaclass import create_delegator, create_super_delegator
%>

## Add lib includes
% for lib in lib_includes:
#include <${lib}>
% endfor

## Add local includes
% for lib in local_includes:
#include "${lib}"
% endfor

#include "${obj_name}.h"

## Process static methods
<%
static_methods = [m.from_myriad_func(m, obj_name + "_" + m.ident) for m in myriad_methods.values()]
%>

## Print methods forward declarations
% for mtd in static_methods:
static ${mtd.stringify_decl()};
% endfor

## Global static variables
% for module_var in myriad_module_vars.values():
    % if type(module_var) is str:
${module_var}
    % else:
${module_var.stringify_decl()};
    % endif
% endfor

## Method definitions
% for mtd in static_methods:
${mtd.stringify_decl()}
{
${mtd.stringify_def()}
}

% endfor

## Method delegators
% for method in own_methods:
<%
    delg = create_delegator(method, cls_name)
    super_delg = create_super_delegator(delg, cls_name)
%>
${delg.stringify_def()}

${super_delg.stringify_def()}
% endfor

## Top-level init function
${init_fun}

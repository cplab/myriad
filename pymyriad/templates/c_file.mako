## Python imports as a module-level block
<%!
    import myriad_types
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

## Print methods forward declarations
% for method in myriad_methods:
static ${obj_name}_${method.stringify_decl()};
% endfor

## Method definitions
% for method in myriad_methods:
${method.stringify_decl()}
{
    ${method.stringify_def()}
}
% endfor

## Method delegators


## Top-level functions
## % for fun in functions:
## ${fun.stringify_decl()}
## {
##     ${fun.stringify_def()}
## }
## % endfor
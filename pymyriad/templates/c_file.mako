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
% for method in methods.values():
    % for i_method in method.instance_methods.values():
${i_method.stringify_decl()};
    % endfor
% endfor

## Print top-level module variables
% for module_var in module_vars.values():
    % if type(module_var) is str:
${module_var}
    % else:
        % if module_var.init is not None:
${module_var.stringify_decl()} = ${module_var.init};
        % else:
${module_var.stringify_decl()};
        % endif
    % endif
% endfor

## Method definitions
% for method in methods.values():
    % for i_method in method.instance_methods.values():
${i_method.stringify_decl()}
{
    ${i_method.fun_def}
}
    % endfor

## Use this trick to force rendering before printing the buffer
${method.delg_template.render() or method.delg_template.buffer}

${method.super_delg_template.render() or method.super_delg_template.buffer}
% endfor

## Top-level functions
% for fun in functions.values():
${fun.stringify_decl()}
{
    ${fun.fun_def}
}
% endfor
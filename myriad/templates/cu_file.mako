## Add lib includes
% for lib in lib_includes:
#include <${lib}>
% endfor

## Add local includes
% for lib in local_includes:
#include "${lib}"
% endfor

## Include ourselves and our children
#include "${obj_name}.cuh"
% for subclass in our_subclasses:
#include "${subclass.__name__}.cuh"
% endfor

## Process instance methods
<%
instance_methods = [m.from_myriad_func(m, obj_name + "_" + m.ident) for m in myriad_methods.values()]
%>

## Global vtables - pre-computed for MyriadObject
% for method in own_methods:

% if obj_name != "MyriadObject":
#ifdef CUDA
__device__ ${method.typedef_name} ${obj_name}_${method.ident}_devp = ${obj_name}_${method.ident};
#endif

#ifdef CUDA
__constant__ const ${method.typedef_name} ${method.ident}_vtable[NUM_CU_CLASS] = { NULL };
#else
% endif
const ${method.typedef_name} ${method.ident}_vtable[NUM_CU_CLASS] = {
    ## For each class in all Myriad classes, use NULL if it's a non-child class and if
    ## the subclass has overwritten the method. Otherwise, use the subclass' version
    ## of our method
    % for cclass in myriad_classes:
        % if cclass.obj_name == obj_name or cclass in our_subclasses:
            % if method.ident not in cclass.myriad_methods:
            &${obj_name}_${method.ident},
            % else:
            &${cclass.obj_name}_${method.ident},
            % endif
        % else:
           NULL,
        % endif
    % endfor
};
% if obj_name != "MyriadObject":
#endif
% endif
% endfor

## Method definitions
% for mtd in instance_methods:
${mtd.stringify_decl()}
{
${mtd.stringify_def()}
}
% endfor

## Method delegators
% for delg, super_delg in own_method_delgs:
${delg.stringify_def()}

${super_delg.stringify_def()}
% endfor

## Top-level init functions
${init_functions}

## Python imports as a module-level block
<%!
    from context import myriad
    from myriad.myriad_metaclass import create_delegator, create_super_delegator
%>

% for lib in cuda_lib_includes:
#include <${lib}>
% endfor

#ifdef __cplusplus
extern "C" {
#endif

% for lib in cuda_local_includes:
#include "${lib}"
% endfor
#include "${obj_name}.cuh"

#ifdef __cplusplus
}
#endif

__constant__ __device__ struct ${cls_name}* ${obj_name}_dev_t = NULL;
__constant__ __device__ struct ${cls_name}* ${cls_name}_dev_t = NULL;

## Method definitions
% for mtd in own_methods:
__device__ ${mtd.stringify_cuda_decl()}
{
${mtd.stringify_def()}
}

__device__ ${mtd.typedef_name} cuda_${mtd.typedef_name} = cuda_${mtd.ident};

% endfor

## Method delegators
## % for method in own_methods:
## <%
##     delg = create_delegator(method, cls_name)
##     super_delg = create_super_delegator(delg, cls_name)
## %>
## ${delg.stringify_def()}
## ${super_delg.stringify_def()}

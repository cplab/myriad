## Add include guards
<% include_guard = obj_name.upper() + "_CUH" %>
#ifndef ${include_guard}
#define ${include_guard}

#ifdef CUDA

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

## Add local includes
% for lib in local_includes:
#include "${lib}"
% endfor

extern __constant__ __device__ struct ${cls_name}* ${obj_name}_dev_t;
extern __constant__ __device__ struct ${cls_name}* ${cls_name}_dev_t;

% for fun in functions.values():
<%
    tmp_fun = fun.copy_init(ident="cuda_" + fun.ident)
    context.write("extern __device__ " + tmp_fun.stringify_decl() + ";")
%>
% endfor

#endif // IFDEF CUDA

#endif
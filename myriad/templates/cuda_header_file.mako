## Add include guards
<% include_guard = obj_name.upper() + "_CUH" %>
#ifndef ${include_guard}
#define ${include_guard}

#ifdef CUDA

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

## Add local includes
#ifdef __cplusplus
extern "C" {
#endif

% for lib in local_includes:
#include "${lib}"
% endfor

#ifdef __cplusplus
}
#endif

extern __constant__ __device__ struct ${cls_name}* ${obj_name}_dev_t;
extern __constant__ __device__ struct ${cls_name}* ${cls_name}_dev_t;

## TODO: Add __device__ function pointer variables (using typedef'd type for myriad method)

## TODO: Add cuda_* versions of delegator function

#endif // IFDEF CUDA

#endif
#ifndef MYRIADOBJECT_CUH
#define MYRIADOBJECT_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MyriadObject.h"

// On-device reference pointers
extern __constant__ __device__ struct MyriadClass* MyriadObject_dev_t;
extern __constant__ __device__ struct MyriadClass* MyriadClass_dev_t;

/////////////////////////////////////
// Object management and Selectors //
/////////////////////////////////////

extern __device__ const void* cuda_myriad_class_of(const void* _self);

extern __device__ size_t cuda_myriad_size_of(const void* self);

extern __device__ int cuda_myriad_is_a(const void* _self, const struct MyriadClass* m_class);

extern __device__ int cuda_myriad_is_of(
    const void* _self,
    const struct MyriadClass* m_class
);

///////////////////////////////
// Super and related methods //
///////////////////////////////

extern __device__ const void* cuda_myriad_super(const void* _self);


#endif

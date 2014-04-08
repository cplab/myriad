/**
   @file    MyriadObject.cuh
 
   @brief   CUDA GPU MyriadObject class definition file.
 
   @details Defines the Myriad object system on the GPU
 
   @author  Pedro Rittner
 
   @date    April 7, 2014
 */
#ifndef MYRIADOBJECT_CUH
#define MYRIADOBJECT_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MyriadObject.h"

// On-device reference pointers

//! On-GPU reference pointer to MyriadObject class prototype
extern __constant__ __device__ struct MyriadClass* MyriadObject_dev_t;

//! On-GPU reference pointer to MyriadClass class prototype
extern __constant__ __device__ struct MyriadClass* MyriadClass_dev_t;

/////////////////////////////////////
// Object management and Selectors //
/////////////////////////////////////

//! On-GPU implementation of myriad_class_of @see myriad_class_of
extern __device__ const void* cuda_myriad_class_of(const void* _self);

//! On-GPU implementation of myriad_size_of @see myriad_size_of
extern __device__ size_t cuda_myriad_size_of(const void* self);

//! On-GPU implementation of myriad_is_a @see myriad_is_a
extern __device__ int cuda_myriad_is_a(
	const void* _self,
	const struct MyriadClass* m_class
);

//! On-GPU implementation of myriad_is_of @see myriad_is_of
extern __device__ int cuda_myriad_is_of(
    const void* _self,
    const struct MyriadClass* m_class
);

///////////////////////////////
// Super and related methods //
///////////////////////////////

//! On-GPU implementation of myriad_super @see myriad_super
extern __device__ const void* cuda_myriad_super(const void* _self);

#endif

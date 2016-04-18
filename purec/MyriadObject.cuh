/**
   @file    MyriadObject.cuh
 
   @brief   CUDA GPU MyriadObject class definition file.
 
   @details Defines the Myriad object system on the GPU
 
   @author  Pedro Rittner
 
   @date    April 7, 2014
 */
#ifndef MYRIADOBJECT_CUH
#define MYRIADOBJECT_CUH

#ifdef CUDA

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

#define cuda_myriad_class_of(self) ((const struct MyriadObject*) self)->m_class

#define cuda_myriad_size_of(self) ((const struct MyriadObject*) self)->m_class->size

#define cuda_myriad_is_a(self, mclass) self && ((const struct MyriadObject*) self)->m_class == mclass;

//! On-GPU implementation of myriad_is_of @see myriad_is_of
extern __device__ int cuda_myriad_is_of(
    const void* _self,
    const struct MyriadClass* m_class
);


#define cuda_myriad_super(self) ((const struct MyriadClass*) self)->super

#endif  // CUDA 
#endif  // MYRIADOBJECT_CUH

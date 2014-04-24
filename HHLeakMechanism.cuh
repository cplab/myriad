/**
   @file    HHLeakMechanism.cuh
 
   @brief   Hodgkin-Huxley Leak Mechanism CUDA definition file.
 
   @details Defines the Hodgkin-Huxley Leak Mechanism CUDA specification for Myriad
 
   @author  Pedro Rittner
 
   @date    April 9, 2014
 */
#ifndef HHLEAKMECHANISM_CUH
#define HHLEAKMECHANISM_CUH

#ifdef CUDA

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MyriadObject.cuh"
#include "Mechanism.cuh"

#include "MyriadObject.h"
#include "Mechanism.h"
#include "HHLeakMechanism.h"

//! On-GPU reference pointer to Mechanism class prototype
extern __device__ __constant__ struct HHLeakMechanism* HHLeakMechanism_dev_t;

//! On-GPU reference pointer to MechanismClass class prototype
extern __device__ __constant__ struct HHLeakMechanismClass* HHLeakMechanismClass_dev_t;

// ----------------------------------------

//! On-GPU reference pointer to Mechanism function implementation
extern __device__ mech_fun_t HHLeakMechanism_mech_fxn_t;

extern __device__ double HHLeakMechanism_cuda_mech_fun(
    void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	);

#endif
#endif

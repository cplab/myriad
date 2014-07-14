/**
   @file    HHLeakMechanism.cu
 
   @brief   Hodgkin-Huxley Leak Mechanism CUDA implementation file.
 
   @details Defines the Hodgkin-Huxley Leak Mechanism CUDA implementation for Myriad
 
   @author  Pedro Rittner
 
   @date    April 23, 2014
 */
#include <stdio.h>

#include <cuda_runtime.h>

extern "C"
{
    #include "myriad_debug.h"
	#include "MyriadObject.h"
    #include "Compartment.h"
	#include "HHSomaCompartment.h"
	#include "Mechanism.h"
	#include "HHLeakMechanism.h"
}

#include "HHSomaCompartment.cuh"
#include "HHLeakMechanism.cuh"

__device__ __constant__ struct HHLeakMechanism* HHLeakMechanism_dev_t;
__device__ __constant__ struct HHLeakMechanismClass* HHLeakMechanismClass_dev_t;

__device__ double HHLeakMechanism_cuda_mech_fun(
    void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	)
{
	const struct HHLeakMechanism* self = (const struct HHLeakMechanism*) _self;
	const struct HHSomaCompartment* c1 = (const struct HHSomaCompartment*) pre_comp;
	const struct HHSomaCompartment* c2 = (const struct HHSomaCompartment*) post_comp;

	//	No extracellular compartment. Current simply "disappears".
	if (c1 == NULL || c1 == c2)
	{
		return -self->g_leak * (c1->vm[curr_step-1] - self->e_rev);
	}else{
		// @TODO Figure out how to do extracellular compartment calc.
		return 0.0;
	}
}

__device__ mech_fun_t HHLeakMechanism_mech_fxn_t = HHLeakMechanism_cuda_mech_fun;

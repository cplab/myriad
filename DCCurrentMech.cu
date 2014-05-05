/**
   @file    DCCurrentMech.cu
 
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
	#include "Mechanism.h"
	#include "DCCurrentMech.h"
}

#include "DCCurrentMech.cuh"

__device__ __constant__ struct DCCurrentMech* DCCurrentMech_dev_t;
__device__ __constant__ struct DCCurrentMechClass* DCCurrentMechClass_dev_t;

__device__ double DCCurrentMech_cuda_mech_fun(
    void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	)
{
	const struct DCCurrentMech* self = (const struct DCCurrentMech*) _self;

	return (curr_step >= self->t_start && curr_step <= self->t_stop) ? self->amplitude : 0.0;
}

__device__ mech_fun_t DCCurrentMech_mech_fxn_t = DCCurrentMech_cuda_mech_fun;

/**
   @file    HHKCurrMechanism.cu
 
   @brief   Hodgkin-Huxley Potassium Mechanism CUDA implementation file.
 
   @details Defines the Hodgkin-Huxley Potassium Mechanism CUDA implementation for Myriad
 
   @author  Pedro Rittner
 
   @date    April 23, 2014
 */
#include <stdio.h>

#include <cuda_runtime.h>

extern "C"
{
	#include "MyriadObject.h"
    #include "Compartment.h"
	#include "HHSomaCompartment.h"
	#include "Mechanism.h"
	#include "HHKCurrMechanism.h"
}

#include "HHSomaCompartment.cuh"
#include "HHKCurrMechanism.cuh"

__device__ __constant__ struct HHKCurrMechanism* HHKCurrMechanism_dev_t;
__device__ __constant__ struct HHKCurrMechanismClass* HHKCurrMechanismClass_dev_t;

__device__ double HHKCurrMechanism_cuda_mech_fun(void* _self,
                                                 void* pre_comp,
                                                 void* post_comp,
                                                 const double dt,
                                                 const double global_time,
                                                 const uint64_t curr_step)
{
	struct HHKCurrMechanism* self = (struct HHKCurrMechanism*) _self;
	const struct HHSomaCompartment* c1 = (const struct HHSomaCompartment*) pre_comp;
	const struct HHSomaCompartment* c2 = (const struct HHSomaCompartment*) post_comp;

	//	Channel dynamics calculation
	const double pre_vm = c1->vm[curr_step-1];
    const double alpha_n = (-0.01 * (pre_vm + 10.0)) / (exp((pre_vm+10.0)/-10.0) - 1.0);
    const double beta_n  = 0.125 * exp(pre_vm/-80.);

    self->hh_n = dt*(alpha_n*(1-self->hh_n) - beta_n*self->hh_n) + self->hh_n;

	//	No extracellular compartment. Current simply "disappears".
	if (c2 == NULL || c1 == c2)
	{
		//	I_K = g_K * hh_n^4 * (Vm[t-1] - e_K)
		return -self->g_k * self->hh_n * self->hh_n * self->hh_n *
				self->hh_n * (pre_vm - self->e_k);
	}else{
		// @TODO Figure out how to do extracellular compartment calc.
		return NAN;
	}
}

__device__ mech_fun_t HHKCurrMechanism_mech_fxn_t = HHKCurrMechanism_cuda_mech_fun;

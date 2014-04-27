/**
   @file    HHNaCurrMechanism.cu
 
   @brief   Hodgkin-Huxley Sodium Mechanism CUDA implementation file.
 
   @details Defines the Hodgkin-Huxley Sodium Mechanism CUDA implementation for Myriad
 
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
	#include "HHNaCurrMechanism.h"
}

#include "HHSomaCompartment.cuh"
#include "HHNaCurrMechanism.cuh"

__device__ __constant__ struct HHNaCurrMechanism* HHNaCurrMechanism_dev_t;
__device__ __constant__ struct HHNaCurrMechanismClass* HHNaCurrMechanismClass_dev_t;

__device__ double HHNaCurrMechanism_cuda_mech_fun(
    void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	)
{
	struct HHNaCurrMechanism* self = (struct HHNaCurrMechanism*) _self;
	const struct HHSomaCompartment* c1 = (const struct HHSomaCompartment*) pre_comp;
	const struct HHSomaCompartment* c2 = (const struct HHSomaCompartment*) post_comp;

	// Channel dynamics calculation
	const double pre_vm = c1->soma_vm[curr_step-1];

	// @TODO: Magic numbers should be extracted out as defines
	const double alpha_m = (0.32*(pre_vm+45.0)) / (1 - exp(-(pre_vm+45.0)/4.0));
	const double beta_m =  (-0.28*(pre_vm+18.0)) / (1 - exp((pre_vm + 18.0)/5.0));
	const double alpha_h = (0.128) / (exp((pre_vm+41.0)/18.0));
	const double beta_h = 4.0 / (1 + exp(-(pre_vm + 18.0)/5.0));

    self->hh_m = dt*( (alpha_m*(1.0-self->hh_m)) - beta_m*self->hh_m) + self->hh_m;
    self->hh_h = dt*( (alpha_h*(1.0-self->hh_h)) - beta_h*self->hh_h) + self->hh_h;

	// No extracellular compartment. Current simply "disappears".
	if (c2 == NULL || c1 == c2)
	{
		// I = g_Na * hh_m^3 * hh_h * (Vm[t-1] - e_rev)
		return -self->g_na * self->hh_m*self->hh_m*self->hh_m * self->hh_h* (pre_vm - self->e_na);
	}else{
		// @TODO Figure out how to do extracellular compartment calc.
		return NAN;
	}	
}

__device__ mech_fun_t HHNaCurrMechanism_mech_fxn_t = HHNaCurrMechanism_cuda_mech_fun;

#endif

/**
   @file    HHSpikeGABAACurrMechanism.cu
 
   @brief   Hodgkin-Huxley Spike-mediated GABA-a Mechanism CUDA implementation file.
 
   @details Defines the Hodgkin-Huxley Spike-mediated GABA-a Mechanism CUDA implementation for Myriad
 
   @author  Pedro Rittner
 
   @date    June 2, 2014
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
	#include "HHSpikeGABAAMechanism.h"
}

#include "HHSomaCompartment.cuh"
#include "HHSpikeGABAAMechanism.cuh"

__device__ __constant__ struct HHSpikeGABAAMechanism* HHSpikeGABAAMechanism_dev_t;
__device__ __constant__ struct HHSpikeGABAAMechanismClass* HHSpikeGABAAMechanismClass_dev_t;

__device__ double HHSpikeGABAAMechanism_cuda_mech_fun(
    void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	)
{
	struct HHSpikeGABAAMechanism* self = (struct HHSpikeGABAAMechanism*) _self;
	const struct HHSomaCompartment* c1 = (const struct HHSomaCompartment*) pre_comp;
	const struct HHSomaCompartment* c2 = (const struct HHSomaCompartment*) post_comp;

	//	Channel dynamics calculation
    const double pre_pre_vm = (curr_step > 1) ? c1->soma_vm[curr_step-2] : INFINITY;
	const double pre_vm = c1->soma_vm[curr_step-1];
	const double post_vm = c2->soma_vm[curr_step-1];
    
    // If we just fired
    if (pre_vm > self->prev_vm_thresh && pre_pre_vm < self->prev_vm_thresh)
    {
        self->t_fired = global_time;
    }

    if (self->t_fired != -INFINITY)
    {
        const double g_s = exp(-(global_time - self->t_fired) / self->tau_beta) - 
            exp(-(global_time - self->t_fired) / self->tau_alpha);
        const double I_GABA = self->norm_const * -self->g_max * g_s * (post_vm - self->gaba_rev);
        return I_GABA;        
    } else {
        return 0.0;
    }
}

__device__ mech_fun_t HHSpikeGABAAMechanism_mech_fxn_t = HHSpikeGABAAMechanism_cuda_mech_fun;

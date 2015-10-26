/**
 *   @file    HHNaCurrMechanism.cu
 *
 * @brief   Hodgkin-Huxley Sodium Mechanism CUDA implementation file.
 *
 * @details Defines the Hodgkin-Huxley Sodium Mechanism CUDA implementation for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    April 23, 2014
 */
#include <stdint.h>

#include <cuda_runtime.h>

extern "C"
{
	#include "MyriadObject.h"
    #include "Compartment.h"
	#include "HHSomaCompartment.h"
	#include "Mechanism.h"
	#include "HHGradedGABAAMechanism.h"
}

#include "HHSomaCompartment.cuh"
#include "HHGradedGABAAMechanism.cuh"

__device__ __constant__ struct HHGradedGABAAMechanism* HHGradedGABAAMechanism_dev_t;
__device__ __constant__ struct HHGradedGABAAMechanismClass* HHGradedGABAAMechanismClass_dev_t;

__device__ double HHGradedGABAAMechanism_cuda_mech_fun(void* _self,
                                                       void* pre_comp,
                                                       void* post_comp,
                                                       const double global_time,
                                                       const uint64_t curr_step)
{
	struct HHGradedGABAAMechanism* self = (struct HHGradedGABAAMechanism*) _self;
	const struct HHSomaCompartment* c1 = (const struct HHSomaCompartment*) pre_comp;
	const struct HHSomaCompartment* c2 = (const struct HHSomaCompartment*) post_comp;

	//	Channel dynamics calculation
	const double pre_vm = c1->vm[curr_step-1];
	const double post_vm = c2->vm[curr_step-1];
	const double prev_g_s = self->g_s[curr_step-1];

	const double fv = 1.0 / (1.0 + exp((pre_vm - self->theta)/-self->sigma));
	self->g_s[curr_step] += DT * (self->tau_alpha * fv * (1.0 - prev_g_s) - self->tau_beta * prev_g_s);

	return -self->g_max * prev_g_s * (post_vm - self->gaba_rev);
}

__device__ mech_fun_t HHGradedGABAAMechanism_mech_fxn_t = HHGradedGABAAMechanism_cuda_mech_fun;

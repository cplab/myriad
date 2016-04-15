#include <stdint.h>

#include <cuda_runtime.h>

extern "C"
{
	#include "MyriadObject.h"
    #include "Compartment.h"
	#include "HHSomaCompartment.h"
}

#include "Compartment.cuh"
#include "Mechanism.cuh"
#include "HHSomaCompartment.cuh"

__device__ void HHSomaCompartment_cuda_simul_fxn(void* _self,
                                                 void** network,
                                                 const double global_time,
                                                 const uint64_t curr_step)
{
	struct HHSomaCompartment* self = (struct HHSomaCompartment*) _self;

	double I_sum = 0.0;

	// Calculate mechanism contribution to current term
	for (uint64_t i = 0; i < self->_.num_mechs; i++)
	{
		struct Mechanism* curr_mech = (struct Mechanism*) self->_.my_mechs[i];
		struct Compartment* pre_comp = (struct Compartment*) network[curr_mech->source_id];
		
		//TODO: Make this conditional on specific Mechanism types
		//if (curr_mech->fx_type == CURRENT_FXN)
		I_sum += cuda_mechanism_fxn(curr_mech, pre_comp, self, global_time, curr_step);
	}

	//	Calculate new membrane voltage: (dVm) + prev_vm
	self->vm[curr_step] = (DT * (I_sum) / (self->cm)) + self->vm[curr_step - 1];

	return;
}

__device__ compartment_simul_fxn_t HHSomaCompartment_simul_fxn_t = HHSomaCompartment_cuda_simul_fxn;

__device__ __constant__ struct HHSomaCompartmentClass* HHSomaCompartmentClass_dev_t;
__device__ __constant__ struct HHSomaCompartment* HHSomaCompartment_dev_t;

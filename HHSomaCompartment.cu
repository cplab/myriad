#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <cuda_runtime.h>

extern "C"
{
    #include "myriad_debug.h"
	#include "MyriadObject.h"
    #include "Compartment.h"
	#include "HHSomaCompartment.h"
}

#include "HHSomaCompartment.cuh"


__device__ void HHSomaCompartment_cuda_simul_fxn(
	void* _self,
	void** network,
	const double dt,
	const double global_time,
	const unsigned int curr_time
	)
{
	struct HHSomaCompartment* self = (struct HHSomaCompartment*) _self;
	printf("I'm HH %u, and I have %u mechanisms. My vm_len is %u\n", self->_.id, self->_.num_mechs, self->soma_vm_len);
	return;
}

__device__ compartment_simul_fxn_t HHSomaCompartment_simul_fxn_t = HHSomaCompartment_cuda_simul_fxn;

__device__ __constant__ struct HHSomaCompartmentClass* HHSomaCompartmentClass_dev_t;
__device__ __constant__ struct HHSomaCompartment* HHSomaCompartment_dev_t;

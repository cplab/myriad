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
}

#include "MyriadObject.cuh"
#include "Compartment.cuh"

__device__ __constant__ struct Compartment* Compartment_dev_t;
__device__ __constant__ struct CompartmentClass* CompartmentClass_dev_t;

__device__ void Compartment_cuda_simul_fxn(
	void* _self,
	void** network,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct Compartment* self = (const struct Compartment*) _self;
	printf("My id is %u\n", self->id);
	printf("My num_mechs is %u\n", self->NUM_MECHS);
	return;
}

__device__ void cuda_simul_fxn(
	void* _self,
	void** network,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct CompartmentClass* m_class =
		(const struct CompartmentClass*) cuda_myriad_class_of((void*) _self);

	return m_class->m_comp_fxn(_self, network, dt, global_time, curr_step);
}

__device__ compartment_simul_fxn_t Compartment_cuda_compartment_fxn_t = Compartment_cuda_simul_fxn;

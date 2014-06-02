#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <cuda_runtime.h>

extern "C"
{
    #include "myriad_debug.h"
    #include "MyriadObject.h"
    #include "Mechanism.h"
}

#include "MyriadObject.cuh"
#include "Mechanism.cuh"

__device__ __constant__ struct Mechanism* Mechanism_dev_t = NULL;
__device__ __constant__ struct MechanismClass* MechanismClass_dev_t = NULL;

__device__ double Mechanism_cuda_mechanism_fxn(
	void* _self,
    void* pre_comp,
    void* post_comp,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct Mechanism* self = (const struct Mechanism*) _self;
	printf("My source id is %u\n", self->source_id);
	return 0.0;
}

__device__ double cuda_mechanism_fxn(
	void* _self,
    void* pre_comp,
    void* post_comp,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct MechanismClass* m_class = (const struct MechanismClass*) cuda_myriad_class_of(_self);

	return m_class->m_mech_fxn(_self, pre_comp, post_comp, dt, global_time, curr_step);
}

__device__ mech_fun_t Mechanism_cuda_mechanism_fxn_t = Mechanism_cuda_mechanism_fxn;

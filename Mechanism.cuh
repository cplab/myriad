#ifndef MECHANISM_CUH
#define MECHANISM_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MyriadObject.cuh"

#include "MyriadObject.h"
#include "Mechanism.h"

extern __device__ __constant__ struct Mechanism* Mechanism_dev_t;
extern __device__ __constant__ struct MechanismClass* MechanismClass_dev_t;

// ----------------------------------------

extern __device__ mech_fun_t Mechanism_cuda_mechanism_fxn_t;

extern __device__ double cuda_mechanism_fxn(
	void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
);


#endif

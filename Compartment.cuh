#ifndef COMPARTMENT_CUH
#define COMPARTMENT_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MyriadObject.cuh"

#include "MyriadObject.h"
#include "Compartment.h"

extern __device__ __constant__ struct Compartment* Compartment_dev_t;
extern __device__ __constant__ struct CompartmentClass* CompartmentClass_dev_t;

// ----------------------------------------

extern __device__ compartment_simul_fxn_t Compartment_cuda_compartment_fxn_t;

extern __device__ void cuda_simul_fxn(
	void* _self,
	void** network,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	);

#endif

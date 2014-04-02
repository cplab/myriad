#ifndef HHSOMACOMPARTMENT_CUH
#define HHSOMACOMPARTMENT_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MyriadObject.cuh"
#include "Compartment.cuh"

#include "MyriadObject.h"
#include "Compartment.h"
#include "HHSomaCompartment.h"

extern __device__ __constant__ struct HHSomaCompartmentClass* HHSomaCompartmentClass_dev_t;
extern __device__ __constant__ struct HHSomaCompartment* HHSomaCompartment_dev_t;

extern __device__ compartment_simul_fxn_t HHSomaCompartment_simul_fxn_t;

extern __device__ void HHSomaCompartment_cuda_simul_fxn(
	void* _self,
	void** network,
	const double dt,
	const double global_time,
	const unsigned int curr_step
);


#endif

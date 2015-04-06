/**
 * @file    HHSpikeGABAAMechanism.cuh
 *
 * @brief   Hodgkin-Huxley Sodium Mechanism CUDA definition file.
 *
 * @details Defines the Hodgkin-Huxley spiked-basde GABA-a Mechanism CUDA
 * specification for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    April 9, 2014
 */
#ifndef HHNSPIKEGABAAMECHANISM_CUH
#define HHNSPIKEGABAAMECHANISM_CUH

#ifdef CUDA

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MyriadObject.cuh"
#include "Mechanism.cuh"

#include "MyriadObject.h"
#include "Mechanism.h"
#include "HHSpikeGABAAMechanism.h"

//! On-GPU reference pointer to Mechanism class prototype
extern __device__ __constant__ struct HHSpikeGABAAMechanism* HHSpikeGABAAMechanism_dev_t;

//! On-GPU reference pointer to MechanismClass class prototype
extern __device__ __constant__ struct HHSpikeGABAAMechanismClass* HHSpikeGABAAMechanismClass_dev_t;

// ----------------------------------------

//! On-GPU reference pointer to Mechanism function implementation
extern __device__ mech_fun_t HHSpikeGABAAMechanism_mech_fxn_t;

extern __device__ double HHSpikeGABAAMechanism_cuda_mech_fun(void* _self,
                                                              void* pre_comp,
                                                              void* post_comp,
                                                              const double global_time,
                                                              const uint64_t curr_step);

#endif /* CUDA */
#endif /* HHNSPIKEGABAAMECHANISM_CUH */

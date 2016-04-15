/**
 * @file    HHGradedGABAAMechanism.cuh
 *
 * @brief   Hodgkin-Huxley Sodium Mechanism CUDA definition file.
 *
 * @details Defines the Hodgkin-Huxley Sodium Mechanism CUDA specification for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    April 9, 2014
 */
#ifndef HHNACURRMECHANISM_CUH
#define HHNACURRMECHANISM_CUH

#ifdef CUDA

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MyriadObject.cuh"
#include "Mechanism.cuh"

#include "MyriadObject.h"
#include "Mechanism.h"
#include "HHGradedGABAAMechanism.h"

//! On-GPU reference pointer to Mechanism class prototype
extern __device__ __constant__ struct HHGradedGABAAMechanism* HHGradedGABAAMechanism_dev_t;

//! On-GPU reference pointer to MechanismClass class prototype
extern __device__ __constant__ struct HHGradedGABAAMechanismClass* HHGradedGABAAMechanismClass_dev_t;

// ----------------------------------------

//! On-GPU reference pointer to Mechanism function implementation
extern __device__ mech_fun_t HHGradedGABAAMechanism_mech_fxn_t;

extern __device__ double HHGradedGABAAMechanism_cuda_mech_fun(void* _self,
                                                              void* pre_comp,
                                                              void* post_comp,
                                                              const double dt,
                                                              const double global_time,
                                                              const uint_fast32_t curr_step);

#endif /* CUDA */
#endif /* HHNACURRMECHANISM_CUH */

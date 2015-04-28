/**
 * @file    DCCurrentMech.cuh
 *
 * @brief   DC Current Mechanism CUDA definition file.
 *
 * @details Defines the DC Current Mechanism CUDA specification for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    May 5, 2014
 */
#ifndef DCCURRENTMECH_CUH
#define DCCURRENTMECH_CUH

#ifdef CUDA

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MyriadObject.cuh"
#include "Mechanism.cuh"

#include "MyriadObject.h"
#include "Mechanism.h"
#include "DCCurrentMech.h"

//! On-GPU reference pointer to Mechanism class prototype
extern __device__ __constant__ struct DCCurrentMech* DCCurrentMech_dev_t;

//! On-GPU reference pointer to MechanismClass class prototype
extern __device__ __constant__ struct DCCurrentMechClass* DCCurrentMechClass_dev_t;

// ----------------------------------------

//! On-GPU reference pointer to Mechanism function implementation
extern __device__ mech_fun_t DCCurrentMech_mech_fxn_t;

extern __device__ double DCCurrentMech_cuda_mech_fun(void* _self,
                                                     void* pre_comp,
                                                     void* post_comp,
                   
                                                     const double global_time,
                                                     const uint64_t curr_step);

#endif /* CUDA */
#endif /* DCCURRENTMECH_CUH */

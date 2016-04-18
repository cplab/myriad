/**
   @file    Mechanism.cuh
 
   @brief   GPU Mechanism class definition file.
 
   @details Defines the device-side Mechanism class specification for Myriad
 
   @author  Pedro Rittner
 
   @date    April 7, 2014
 */
#ifndef MECHANISM_CUH
#define MECHANISM_CUH

#ifdef CUDA

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MyriadObject.cuh"

#include "MyriadObject.h"
#include "Mechanism.h"

//! On-GPU reference pointer to Mechanism class prototype
extern __device__ __constant__ struct Mechanism* Mechanism_dev_t;

//! On-GPU reference pointer to MechanismClass class prototype
extern __device__ __constant__ struct MechanismClass* MechanismClass_dev_t;

// ----------------------------------------

//! On-GPU reference pointer to Mechanism function implementation
extern __device__ mech_fun_t Mechanism_cuda_mechanism_fxn_t;

/**
 * On-GPU Delegator function for MechanismClass mechansim function method.
 *
 * @param[in]  _self        pointer to extant object instance
 * @param[in]  pre_comp     pointer to the compartment where this mechanism is
 * @param[in]  global_time  current global time of the simulation
 * @param[in]  curr_step    current global time step of the simulation
 *
 * @returns calculated output value of this mechanism for the given timestep
 */
extern __device__ double cuda_mechanism_fxn(void* _self,
                                            void* pre_comp,
                                            void* post_comp,
                                            const double global_time,
                                            const uint_fast32_t curr_step);

#endif /* CUDA */
#endif /* MECHANISM_CUH */

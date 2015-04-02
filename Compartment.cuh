/**
 * @file    Compartment.cuh
 *
 * @brief   GPU Compartment class definition file.
 *
 * @details Defines the device-side Compartment class specification for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    April 7, 2014
 */
#ifndef COMPARTMENT_CUH
#define COMPARTMENT_CUH

#ifdef CUDA

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MyriadObject.cuh"

#include "MyriadObject.h"
#include "Compartment.h"

//! On-GPU reference pointer to Compartment class prototype
extern __device__ __constant__ struct Compartment* Compartment_dev_t;

//! On-GPU reference pointer to CompartmentClass class prototype
extern __device__ __constant__ struct CompartmentClass* CompartmentClass_dev_t;

// ----------------------------------------

//! On-GPU reference pointer to Compartment function implementation
extern __device__ compartment_simul_fxn_t Compartment_cuda_compartment_fxn_t;

/**
 *   On-GPU Delegator function for CompartmentClass mechansim function method.
 *
 * @param[in]  _self        pointer to extant object instance
 * @param[in]  network      list of pointers to other compartments in network
 * @param[in]  dt           timestep of the simulation
 * @param[in]  global_time  current global time of the simulation
 * @param[in]  curr_step    current global time step of the simulation
 */
extern __device__ void cuda_simul_fxn(void* _self,
                                      void** network,
                                      const double dt,
                                      const double global_time,
                                      const uint64_t curr_step);

#endif /* CUDA */

#endif /* COMPARTMENT_CUH */

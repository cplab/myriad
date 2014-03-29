#ifndef COMPARTMENT_H
#define COMPARTMENT_H

#include <stddef.h>

#include "MyriadObject.cuh"

// Forward struct declaration
struct Compartment;
struct CompartmentClass;

// Compartment simulate function definition
typedef void (* compartment_simul_fxn_t) (
	void* _self,
	void** network,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	);

// Expose interface to myriad_new()
extern const void* Compartment;
extern const void* CompartmentClass;

extern __device__ __constant__ struct Compartment* Compartment_dev_t;
extern __device__ __constant__ struct CompartmentClass* CompartmentClass_dev_t;

extern void simul_fxn(
	void* _self,
	void** network,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	);

extern __device__ void cuda_simul_fxn(
	void* _self,
	void** network,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	);

struct Compartment
{
	const struct MyriadObject _; // Compartment : MyriadObject
	unsigned int id;
};

struct CompartmentClass
{
	const struct MyriadClass _; // CompartmentClass : MyriadClass
	compartment_simul_fxn_t m_comp_fxn;
};

void initCompartment(const int init_cuda);

#endif

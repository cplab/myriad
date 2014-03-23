#ifndef MECHANISM_H
#define MECHANISM_H

#include <stdlib.h>
#include <stddef.h>

#include "MyriadObject.cuh"

// Mechanism function typedef
typedef double (* mech_fun_t) (
	void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
);

extern const void* Mechanism; // myriad_new(Mechanism, ...);
extern const void* MechanismClass;

extern __device__ __constant__ struct Mechanism* Mechanism_dev_t;
extern __device__ __constant__ struct MechanismClass* MechanismClass_dev_t;

// -----------------------------------------

extern double mechanism_fxn(
	void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
);

extern __device__ double cuda_mechanism_fxn(
	void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
);


// ----------------------------------------

struct Mechanism
{
	const struct MyriadObject _; // Mechanism : MyriadObject
	unsigned int source_id;
};

struct MechanismClass
{
	const struct MyriadClass _; // MechanismClass : MyriadClass
	mech_fun_t m_mech_fxn;
};

// -------------------------------------

void initMechanism(const int init_cuda);

#endif

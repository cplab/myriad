#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "MyriadObject.h"
#include "Mechanism.h"
#include "HHSomaCompartment.h"
#include "HHKCurrMechanism.h"
#include "HHKCurrMechanism.cuh"

///////////////////////////////////////
// HHKCurrMechanism Super Overrides //
///////////////////////////////////////

static void* HHKCurrMechanism_ctor(void* _self, va_list* app)
{
	struct HHKCurrMechanism* self = 
		(struct HHKCurrMechanism*) super_ctor(HHKCurrMechanism, _self, app);
    
	self->g_k = va_arg(*app, double);
	self->e_k = va_arg(*app, double);
	self->hh_n = va_arg(*app, double);

	return self;
}

static double HHKCurrMechanism_mech_fun(
    void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	)
{
	struct HHKCurrMechanism* self = (struct HHKCurrMechanism*) _self;
	const struct HHSomaCompartment* c1 = (const struct HHSomaCompartment*) pre_comp;
	const struct HHSomaCompartment* c2 = (const struct HHSomaCompartment*) post_comp;

	//	Channel dynamics calculation
	const double pre_vm = c1->vm[curr_step-1];

    const double alpha_n = (-0.01 * (pre_vm + 34.)) / (exp((pre_vm+34.0)/-1.) - 1.);
    const double beta_n  = 0.125 * exp((pre_vm + 44.)/-80.);

    self->hh_n += dt*5.*(alpha_n*(1.-self->hh_n) - beta_n*self->hh_n);

	//	No extracellular compartment. Current simply "disappears".
	if (c2 == NULL || c1 == c2)
	{
		//	I_K = g_K * hh_n^4 * (Vm[t-1] - e_K)
		return -self->g_k * self->hh_n * self->hh_n * self->hh_n *
				self->hh_n * (pre_vm - self->e_k);

	}else{
		// @TODO Figure out how to do extracellular compartment calc.
		return NAN;
	}
}

////////////////////////////////////////////
// HHKCurrMechanismClass Super Overrides //
////////////////////////////////////////////

static void* HHKCurrMechanismClass_cudafy(void* _self, int clobber)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct HHKCurrMechanismClass* my_class = (struct HHKCurrMechanismClass*) _self;

		// Make a temporary copy-class because we need to change shit
		struct HHKCurrMechanismClass copy_class = *my_class;
		struct MyriadClass* copy_class_class = (struct MyriadClass*) &copy_class;
	
		// !!!!!!!!! IMPORTANT !!!!!!!!!!!!!!
		// By default we clobber the copy_class_class' superclass with
		// the superclass' device_class' on-GPU address value. 
		// To avoid cloberring this value (e.g. if an underclass has already
		// clobbered it), the clobber flag should be 0.
		if (clobber)
		{
			// TODO: Find a better way to get function pointers for on-card functions
			mech_fun_t my_mech_fun = NULL;
			CUDA_CHECK_RETURN(
				cudaMemcpyFromSymbol(
					(void**) &my_mech_fun,
					(const void*) &HHKCurrMechanism_mech_fxn_t,
					sizeof(void*),
					0,
					cudaMemcpyDeviceToHost
					)
				);
			copy_class._.m_mech_fxn = my_mech_fun;
		
			DEBUG_PRINTF("Copy Class mech fxn: %p\n", my_mech_fun);
		
			const struct MyriadClass* super_class = (const struct MyriadClass*) MechanismClass;
			memcpy((void**) &copy_class_class->super, &super_class->device_class, sizeof(void*));
		}

		// This works because super methods rely on the given class'
		// semi-static superclass definition, not it's ->super attribute.
		// Note that we don't want to clobber, so we set it to 0.
		return super_cudafy(MechanismClass, (void*) &copy_class, 0);
	}
	#else
	{
		return NULL;
	}
	#endif
}

////////////////////////////
// Dynamic Initialization //
////////////////////////////

const void* HHKCurrMechanism;
const void* HHKCurrMechanismClass;

void initHHKCurrMechanism(int init_cuda)
{
	// initCompartment(init_cuda);
	
	if (!HHKCurrMechanismClass)
	{
		HHKCurrMechanismClass =
			myriad_new(
				MechanismClass,
				MechanismClass,
				sizeof(struct HHKCurrMechanismClass),
				myriad_cudafy, HHKCurrMechanismClass_cudafy,
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)HHKCurrMechanismClass, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) HHKCurrMechanismClass)->device_class = 
				(struct MyriadClass*) tmp_mech_c_t;
			
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &HHKCurrMechanismClass_dev_t,
					&tmp_mech_c_t,
					sizeof(struct HHKCurrMechanismClass*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}

	if (!HHKCurrMechanism)
	{
		HHKCurrMechanism =
			myriad_new(
				HHKCurrMechanismClass,
				Mechanism,
				sizeof(struct HHKCurrMechanism),
				myriad_ctor, HHKCurrMechanism_ctor,
				mechanism_fxn, HHKCurrMechanism_mech_fun,
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)HHKCurrMechanism, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) HHKCurrMechanism)->device_class = 
				(struct MyriadClass*) tmp_mech_t;

			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &HHKCurrMechanism_dev_t,
					&tmp_mech_t,
					sizeof(struct HHKCurrMechanism*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
}




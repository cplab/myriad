#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "MyriadObject.h"
#include "Mechanism.h"
#include "HHSomaCompartment.h"
#include "HHNaCurrMechanism.h"
#include "HHNaCurrMechanism.cuh"

///////////////////////////////////////
// HHNaCurrMechanism Super Overrides //
///////////////////////////////////////

static void* HHNaCurrMechanism_ctor(void* _self, va_list* app)
{
	struct HHNaCurrMechanism* self = 
		(struct HHNaCurrMechanism*) super_ctor(HHNaCurrMechanism, _self, app);
    
	self->g_na = va_arg(*app, double);
	self->e_na = va_arg(*app, double);
	self->hh_m = va_arg(*app, double);
	self->hh_h = va_arg(*app, double);

	return self;
}

static double HHNaCurrMechanism_mech_fun(void* _self,
                                         void* pre_comp,
                                         void* post_comp,
 
                                         const double global_time,
                                         const uint64_t curr_step)
{
	struct HHNaCurrMechanism* self = (struct HHNaCurrMechanism*) _self;
	const struct HHSomaCompartment* c1 = (const struct HHSomaCompartment*) pre_comp;
	const struct HHSomaCompartment* c2 = (const struct HHSomaCompartment*) post_comp;

	//	Channel dynamics calculation
	const double pre_vm = c1->vm[curr_step-1];
    
	const double alpha_m = (-0.1*(pre_vm + 35.)) / (EXP(-0.1*(pre_vm+35.)) - 1.);
	const double beta_m =  4. * EXP((pre_vm + 60.) / -18.);
	const double alpha_h = (0.128) / (EXP((pre_vm+41.0)/18.0));
	const double beta_h = 4.0 / (1 + EXP(-(pre_vm + 18.0)/5.0));

	const double minf = (alpha_m/(alpha_m + beta_m));
	self->hh_h += DT* 5. *(alpha_h*(1. - self->hh_h) - (beta_h * self->hh_h));

	//	No extracellular compartment. Current simply "disappears".
	if (c2 == NULL || c1 == c2)
	{
		//	I = g_Na * minf^3 * hh_h * (Vm[t-1] - e_rev)
		const double I_Na = -self->g_na * minf * minf * minf *	self->hh_h *
				(pre_vm - self->e_na);
		return I_Na;

	}else{
		// @TODO Figure out how to do extracellular compartment calc.
		return NAN;
	}
}

////////////////////////////////////////////
// HHNaCurrMechanismClass Super Overrides //
////////////////////////////////////////////

static void* HHNaCurrMechanismClass_cudafy(void* _self, int clobber)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct HHNaCurrMechanismClass* my_class = (struct HHNaCurrMechanismClass*) _self;

		// Make a temporary copy-class because we need to change shit
		struct HHNaCurrMechanismClass copy_class = *my_class;
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
					(const void*) &HHNaCurrMechanism_mech_fxn_t,
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

const void* HHNaCurrMechanism;
const void* HHNaCurrMechanismClass;

void initHHNaCurrMechanism(const bool init_cuda)
{
	// initCompartment(init_cuda);
	
	if (!HHNaCurrMechanismClass)
	{
		HHNaCurrMechanismClass =
			myriad_new(
				MechanismClass,
				MechanismClass,
				sizeof(struct HHNaCurrMechanismClass),
				myriad_cudafy, HHNaCurrMechanismClass_cudafy,
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)HHNaCurrMechanismClass, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) HHNaCurrMechanismClass)->device_class = 
				(struct MyriadClass*) tmp_mech_c_t;
			
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &HHNaCurrMechanismClass_dev_t,
					&tmp_mech_c_t,
					sizeof(struct HHNaCurrMechanismClass*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}

	if (!HHNaCurrMechanism)
	{
		HHNaCurrMechanism =
			myriad_new(
				HHNaCurrMechanismClass,
				Mechanism,
				sizeof(struct HHNaCurrMechanism),
				myriad_ctor, HHNaCurrMechanism_ctor,
				mechanism_fxn, HHNaCurrMechanism_mech_fun,
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)HHNaCurrMechanism, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) HHNaCurrMechanism)->device_class = 
				(struct MyriadClass*) tmp_mech_t;

			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &HHNaCurrMechanism_dev_t,
					&tmp_mech_t,
					sizeof(struct HHNaCurrMechanism*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
}




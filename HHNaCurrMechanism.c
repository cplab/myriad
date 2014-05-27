#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "myriad_debug.h"

#include "MyriadObject.h"
#include "Mechanism.h"
#include "HHSomaCompartment.h"
#include "HHNaCurrMechanism.h"
#include "HHNaCurrMechanism.cuh"

///////////////////////////////////////
// HHNaCurrMechanism Super Overrides //
///////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, HHNACURRMECHANISM_OBJECT, CTOR_FUN_NAME)
//static void* HHNaCurrMechanism_ctor(void* _self, va_list* app)
{
	struct HHNACURRMECHANISM_OBJECT* _self = 
		(struct HHNACURRMECHANISM_OBJECT*) super_ctor(HHNACURRMECHANISM_OBJECT, self, app);
    
	_self->HHNACURRMECHANISM_CHANNEL_CONDUCTANCE = va_arg(*app, double);
	_self->HHNACURRMECHANISM_REVERSAL_POTENTIAL = va_arg(*app, double);
	_self->HHNACURRMECHANISM_HH_M = va_arg(*app, double);
	_self->HHNACURRMECHANISM_HH_H = va_arg(*app, double);

	return self;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(HHNACURRMECHANISM_MECH_FXN_RET, HHNACURRMECHANISM_MECH_FXN_ARGS, HHNACURRMECHANISM_OBJECT, HHNACURRMECHANISM_MECH_FXN_NAME)
/* static double HHNaCurrMechanism_mech_fun(
    void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	)
*/
{
	struct HHNACURRMECHANISM_OBJECT* self = (struct HHNACURRMECHANISM_OBJECT*) _self;
	const struct HHSOMACOMPARTMENT_OBJECT* c1 = (const struct HHSOMACOMPARTMENT_OBJECT*) pre_comp;
	const struct HHSOMACOMPARTMENT_OBJECT* c2 = (const struct HHSOMACOMPARTMENT_OBJECT*) post_comp;

	//	Channel dynamics calculation
	const double pre_vm = c1->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE[curr_step-1];

	const double alpha_m = (-0.1*(pre_vm + 35.)) / (exp(-0.1*(pre_vm+35.)) - 1.) ;
	const double beta_m =  4. * exp((pre_vm + 60.) / -18.);
	const double alpha_h = (0.128) / (exp((pre_vm+41.0)/18.0));
	const double beta_h = 4.0 / (1 + exp(-(pre_vm + 18.0)/5.0));

	const double minf = (alpha_m/(alpha_m + beta_m));
	self->HHNACURRMECHANISM_HH_H += dt* 5. *(alpha_h*(1. - self->hh_h) - (beta_h * self->HHNACURRMECHANISM_HH_H));

	//	No extracellular compartment. Current simply "disappears".
	if (c2 == NULL || c1 == c2)
	{
		//	I = g_Na * minf^3 * hh_h * (Vm[t-1] - e_rev)
		const double I_Na = -self->HHNACURRMECHANISM_CHANNEL_CONDUCTANCE * minf * minf * minf *	self->HHNACURRMECHANISM_HH_H *
				(pre_vm - self->HHNACURRMECHANISM_REVERSAL_POTENTIAL);
		return I_Na;

	}else{
		// @TODO Figure out how to do extracellular compartment calc.
		return NAN; //TODO: GENERICISE THIS!
	}
}

////////////////////////////////////////////
// HHNaCurrMechanismClass Super Overrides //
////////////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, HHNACURRMECHANISM_CLASS, CUDAFY_FUN_NAME)
//static void* HHNaCurrMechanismClass_cudafy(void* _self, int clobber)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct HHNACURRMECHANISM_CLASS* my_class = (struct HHNACURRMECHANISM_CLASS*) _self;

		// Make a temporary copy-class because we need to change shit
		struct HHNACURRMECHANISM_CLASS copy_class = *my_class;
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
					(const void*) &MYRIAD_CAT(HHNACURRMECHANISM_OBJECT, _mech_fxn_t),
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

const void* HHNACURRMECHANISM_OBJECT;
const void* HHNACURRMECHANISM_CLASS;

void initHHNaCurrMechanism(int init_cuda)
{
	// initCompartment(init_cuda);
	
	if (!HHNACURRMECHANISM_CLASS)
	{
		HHNACURRMECHANISM_CLASS =
			myriad_new(
				MechanismClass,
				MechanismClass,
				sizeof(struct HHNACURRMECHANISM_CLASS),
				myriad_cudafy, MYRIAD_CAT(HHNACURRMECHANISM_CLASS, _cudafy),
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)HHNACURRMECHANISM_CLASS, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) HHNACURRMECHANISM_CLASS)->device_class = 
				(struct MyriadClass*) tmp_mech_c_t;
			
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(HHNACURRMECHANISM_CLASS, _dev_t),
					&tmp_mech_c_t,
					sizeof(struct HHNACURRMECHANISM_CLASS*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}

	if (!HHNACURRMECHANISM_OBJECT)
	{
		HHNACURRMECHANISM_OBJECT =
			myriad_new(
				HHNACURRMECHANISM_CLASS,
				Mechanism,
				sizeof(struct HHNACURRMECHANISM_OBJECT),
				myriad_ctor, MYRIAD_CAT(HHNACURRMECHANISM_OBJECT, _ctor),
				mechanism_fxn, MYRIAD_CAT(HHNACURRMECHANISM_OBJECT, _mech_fun),
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)HHNACURRMECHANISM_OBJECT, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) HHNACURRMECHANISM_OBJECT)->device_class = 
				(struct MyriadClass*) tmp_mech_t;

			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(HHNACURRMECHANISM_OBJECT, _dev_t),
					&tmp_mech_t,
					sizeof(struct HHNACURRMECHANISM_OBJECTs*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
}




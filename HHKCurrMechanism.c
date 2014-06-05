#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "myriad_debug.h"

#include "MyriadObject.h"
#include "Mechanism.h"
#include "HHSomaCompartment.h"
#include "HHKCurrMechanism.h"
#include "HHKCurrMechanism.cuh"

///////////////////////////////////////
// HHKCurrMechanism Super Overrides //
///////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, HHKCURRMECHANISM_OBJECT, CTOR_FUN_NAME)
{
	struct HHKCURRMECHANISM_OBJECT* _self = 
		(struct HHKCURRMECHANISM_OBJECT*) SUPERCLASS_CTOR(HHKCURRMECHANISM_OBJECT, self, app);
    
	_self->HHKCURRMECHANISM_CHANNEL_CONDUCTANCE = va_arg(*app, double);
	_self->HHKCURRMECHANISM_REVERE_POTENTIAL = va_arg(*app, double);
	_self->HHKCURRMECHANISM_HH_N = va_arg(*app, double);

	return self;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(MECH_FXN_RET, MECH_FXN_ARGS, HHKCURRMECHANISM_OBJECT, INDIVIDUAL_MECH_FXN_NAME)
{
	struct HHKCURRMECHANISM_OBJECT* self = (struct HHKCURRMECHANISM_OBJECT*) _self;
	const struct HHSOMACOMPARTMENT_OBJECT* c1 = (const struct HHSOMACOMPARTMENT_OBJECT*) pre_comp;
	const struct HHSOMACOMPARTMENT_OBJECT* c2 = (const struct HHSOMACOMPARTMENT_OBJECT*) post_comp;

	//	Channel dynamics calculation
	const double pre_vm = c1->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE[curr_step-1];

    const double alpha_n = (-0.01 * (pre_vm + 34.)) / (exp((pre_vm+34.0)/-1.) - 1.);
    const double beta_n  = 0.125 * exp((pre_vm + 44.)/-80.);

    self->HHKCURRMECHANISM_HH_N += dt*5.*(alpha_n*(1.-self->HHKCURRMECHANISM_HH_N) - beta_n*self->HHKCURRMECHANISM_HH_N);

	//	No extracellular compartment. Current simply "disappears".
	if (c2 == NULL || c1 == c2)
	{
		//	I_K = g_K * hh_n^4 * (Vm[t-1] - e_K)
		return -self->HHKCURRMECHANISM_CHANNEL_CONDUCTANCE * self->HHKCURRMECHANISM_HH_N * self->HHKCURRMECHANISM_HH_N * self->HHKCURRMECHANISM_HH_N *
				self->HHKCURRMECHANISM_HH_N * (pre_vm - self->HHKCURRMECHANISM_REVERE_POTENTIAL);

	}else{
		// @TODO Figure out how to do extracellular compartment calc.
		return NAN; //TODO: genericise this!
	}
}

////////////////////////////////////////////
// HHKCurrMechanismClass Super Overrides ///
////////////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, HHKCURRMECHANISM_CLASS, CUDAFY_FUN_NAME)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct HHKCURRMECHANISM_CLASS* my_class = (struct HHKCURRMECHANISM_CLASS*) _self;

		// Make a temporary copy-class because we need to change shit
		struct HHKCURRMECHANISM_CLASS copy_class = *my_class;
		struct MYRIADOBJECT_CLASS* copy_class_class = (struct MYRIADOBJECT_CLASS*) &copy_class;
	
		// !!!!!!!!! IMPORTANT !!!!!!!!!!!!!!
		// By default we clobber the copy_class_class' superclass with
		// the superclass' device_class' on-GPU address value. 
		// To avoid cloberring this value (e.g. if an underclass has already
		// clobbered it), the clobber flag should be 0.
		if (clobber)
		{
			// TODO: Find a better way to get function pointers for on-card functions
			MECH_FXN_NAME_T my_mech_fun = NULL;
			CUDA_CHECK_RETURN(
				cudaMemcpyFromSymbol(
					(void**) &my_mech_fun,
                    //TODO: Genericise this out
					(const void*) &HHKCurrMechanism_mech_fxn_t,
					sizeof(void*),
					0,
					cudaMemcpyDeviceToHost
					)
				);
			copy_class._.MY_MECHANISM_MECH_CLASS_FXN = my_mech_fun;
		
			DEBUG_PRINTF("Copy Class mech fxn: %p\n", my_mech_fun);
		
			const struct MYRIADOBJECT_CLASS* super_class = (const struct MYRIADOBJECT_CLASS*) MECHANISM_CLASS;
			memcpy((void**) &copy_class_class->SUPERCLASS, &super_class->ONDEVICE_CLASS, sizeof(void*));
		}

		// This works because super methods rely on the given class'
		// semi-static superclass definition, not it's ->super attribute.
		// Note that we don't want to clobber, so we set it to 0.
		return SUPERCLASS_CUDAFY(MECHANISM_CLASS, (void*) &copy_class, 0);
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

const void* HHKCURRMECHANISM_OBJECT;
const void* HHKCURRMECHANISM_CLASS;

MYRIAD_FXN_METHOD_HEADER_GEN_NO_SUFFIX(DYNAMIC_INIT_FXN_RET, DYNAMIC_INIT_FXN_ARGS, HHKCURRMECHANISM_INIT_FXN_NAME)
//void initHHKCurrMechanism(int init_cuda)
{
	// initCompartment(init_cuda);
	
	if (!HHKCURRMECHANISM_CLASS)
	{
		HHKCURRMECHANISM_CLASS =
			myriad_new(
				MECHANISM_CLASS,
				MECHANISM_CLASS,
				sizeof(struct HHKCURRMECHANISM_CLASS),
				myriad_cudafy, MYRIAD_CAT(HHKCURRMECHANISM_CLASS, MYRIAD_CAT(_, CUDAFY_FUN_NAME)),
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)HHKCURRMECHANISM_CLASS, 1);
			// Set our device class to the newly-cudafied class object
			((struct MYRIADOBJECT_CLASS*) HHKCURRMECHANISM_CLASS)->ONDEVICE_CLASS = 
				(struct MYRIADOBJECT_CLASS*) tmp_mech_c_t;
			
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(HHKCURRMECHANISM_CLASS, MYRIAD_CAT(_, DEV_T)),
					&tmp_mech_c_t,
					sizeof(struct HHKCURRMECHANISM_CLASS*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}

	if (!HHKCURRMECHANISM_OBJECT)
	{
		HHKCURRMECHANISM_OBJECT =
			myriad_new(
				HHKCURRMECHANISM_CLASS,
				MECHANISM_OBJECT,
				sizeof(struct HHKCURRMECHANISM_OBJECT),
				myriad_ctor, MYRIAD_CAT(HHKCURRMECHANISM_OBJECT, MYRIAD_CAT(_, CTOR_FUN_NAME)),
				MECH_FXN_NAME_D, MYRIAD_CAT(HHKCURRMECHANISM_OBJECT, MYRIAD_CAT(_, INDIVIDUAL_MECH_FXN_NAME)),
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)HHKCURRMECHANISM_OBJECT, 1);
			// Set our device class to the newly-cudafied class object
			((struct MYRIADOBJECT_CLASS*) HHKCURRMECHANISM_OBJECT)->ONDEVICE_CLASS = 
				(struct MYRIADOBJECT_CLASS*) tmp_mech_t;

			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(HHKCURRMECHANISM_OBJECT, MYRIAD_CAT(_, DEV_T)),
					&tmp_mech_t,
					sizeof(struct HHKCURRMECHANISM_OBJECT*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
}




#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "myriad_debug.h"

#include "MyriadObject.h"
#include "Mechanism.h"
#include "HHSomaCompartment.h"
#include "HHGradedGABAAMechanism.h"
#include "HHGradedGABAAMechanism.cuh"

///////////////////////////////////////
// HHGradedGABAAMechanism Super Overrides //
///////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, HHGRADEDGABAAMECHANISM_OBJECT, CTOR_FUN_NAME)
{
	struct HHGRADEDGABAAMECHANISM_OBJECT* _self = 
		(struct HHGRADEDGABAAMECHANISM_OBJECT*) SUPERCLASS_CTOR(HHGRADEDGABAAMECHANISM_OBJECT, self, app);

	const double HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING_INIT = va_arg(*app, double);
	_self->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING = va_arg(*app, double*);
	_self->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING_LENGTH = va_arg(*app, unsigned int);

	if (_self->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING == NULL && _self->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING_LENGTH > 0)
	{
		_self->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING = (double*) calloc(_self->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING_LENGTH, sizeof(double));
	}
	
	if (_self->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING != NULL)
	{
		_self->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING[0] = HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING_INIT;
	}

	_self->HHGRADEDGABAAMECHANISM_MAX_SYN_CONDUCTANCE = va_arg(*app, double);
	_self->HHGRADEDGABAAMECHANISM_HALF_ACTIVATION_POTENTIAL = va_arg(*app, double);
	_self->HHGRADEDGABAAMECHANISM_MAXIMAL_SLOPE = va_arg(*app, double);
	_self->HHGRADEDGABAAMECHANISM_CHANNEL_OPENING_TIME = va_arg(*app, double);
	_self->HHGRADEDGABAAMECHANISM_CHANNEL_CLOSING_TIME = va_arg(*app, double);
	_self->HHGRADEDGABAAMECHANISM_REVERSAL_POTENTIAL = va_arg(*app, double);

	return self;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(HHGRADEDGABAAMECHANISM_MECH_FXN_RET, HHGRADEDGABAAMECHANISM_MECH_FXN_ARGS, HHGRADEDGABAAMECHANISM_OBJECT, HHGRADEDGABAAMECHANISM_MECH_FXN_NAME)
{
	struct HHGRADEDGABAAMECHANISM_OBJECT* self = (struct HHGRADEDGABAAMECHANISM_OBJECT*) _self;
	const struct HHSOMACOMPARTMENT_OBJECT* c1 = (const struct HHSOMACOMPARTMENT_OBJECT*) pre_comp;
	const struct HHSOMACOMPARTMENT_OBJECT* c2 = (const struct HHSOMACOMPARTMENT_OBJECT*) post_comp;

	//	Channel dynamics calculation
	const double pre_vm = c1->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE[curr_step-1]; 
	const double post_vm = c2->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE[curr_step-1];
	const double prev_g_s = self->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING[curr_step-1];

	const double fv = 1.0 / (1.0 + exp((pre_vm - self->HHGRADEDGABAAMECHANISM_HALF_ACTIVATION_POTENTIAL)/-self->HHGRADEDGABAAMECHANISM_MAXIMAL_SLOPE));
	self->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING[curr_step] += dt * (self->HHGRADEDGABAAMECHANISM_CHANNEL_OPENING_TIME * fv * (1.0 - prev_g_s) - self->HHGRADEDGABAAMECHANISM_CHANNEL_CLOSING_TIME * prev_g_s);

	const double I_GABA = -self->HHGRADEDGABAAMECHANISM_MAX_SYN_CONDUCTANCE * prev_g_s * (post_vm - self->HHGRADEDGABAAMECHANISM_REVERSAL_POTENTIAL);
	return I_GABA;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, HHGRADEDGABAAMECHANISM_OBJECT, CUDAFY_FUN_NAME)
{
	#ifdef CUDA
	{
		const size_t my_size = myriad_size_of(_self);
		struct HHGRADEDGABAAMECHANISM_OBJECT* self = (struct HHGRADEDGABAAMECHANISM_OBJECT*) _self;
		struct HHGRADEDGABAAMECHANISM_OBJECT* self_copy = (struct HHGRADEDGABAAMECHANISM_OBJECT*) calloc(1, my_size);
		
		memcpy(self_copy, self, my_size);

		double* tmp_alias = NULL;
		
		// Make mirror on-GPU array 
		CUDA_CHECK_RETURN(
			cudaMalloc(
				(void**) &tmp_alias,
				self_copy->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING_LENGTH * sizeof(double)
				)
			);

		// Copy contents over to GPU
		CUDA_CHECK_RETURN(
			cudaMemcpy(
				(void*) tmp_alias,
				(void*) self->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING,
				self_copy->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING_LENGTH * sizeof(double),
				cudaMemcpyHostToDevice
				)
			);

		self_copy->HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING = tmp_alias;

		return SUPERCLASS_CUDAFY(HHSOMACOMPARTMENT_OBJECT, self_copy, 0);
	}
	#else
	{
	    return NULL;
    }
	#endif
}

/////////////////////////////////////////////////
// HHGradedGABAAMechanismClass Super Overrides //
/////////////////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, HHGRADEDGABAAMECHANISM_CLASS, CUDAFY_FUN_NAME)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct HHGRADEDGABAAMECHANISM_CLASS* my_class = (struct HHGRADEDGABAAMECHANISM_CLASS*) _self;

		// Make a temporary copy-class because we need to change shit
		struct HHGRADEDGABAAMECHANISM_CLASS copy_class = *my_class;
		struct MYRIADOBJECT_CLASS* copy_class_class = (struct MYRIADOBJECT_CLASS*) &copy_class;
	
		// !!!!!!!!! IMPORTANT !!!!!!!!!!!!!!
		// By default we clobber the copy_class_class' superclass with
		// the superclass' device_class' on-GPU address value. 
		// To avoid cloberring this value (e.g. if an underclass has already
		// clobbered it), the clobber flag should be 0.
		if (clobber)
		{
			// TODO: Find a better way to get function pointers for on-card functions
			MECH_FXN_NAME my_mech_fun = NULL;
			CUDA_CHECK_RETURN(
				cudaMemcpyFromSymbol(
					(void**) &my_mech_fun,
					(const void*) &MYRIAD_CAT(HHGRADEDGABAAMECHANISM_OBJECT, MYRIAD_CAT(_, MECH_FXN_NAME)),
					sizeof(void*),
					0,
					cudaMemcpyDeviceToHost
					)
				);
			copy_class._.m_mech_fxn = my_mech_fun;
		
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

const void* HHGRADEDGABAAMECHANISM_OBJECT;
const void* HHGRADEDGABAAMECHANISM_CLASS;

void initHHGradedGABAAMechanism(int init_cuda)
{
	if (!HHGRADEDGABAAMECHANISM_CLASS)
	{
		HHGRADEDGABAAMECHANISM_CLASS =
			myriad_new(
				MECHANISM_CLASS,
				MECHANISM_CLASS,
				sizeof(struct HHGRADEDGABAAMECHANISM_CLASS),
				myriad_cudafy, MYRIAD_CAT(HHGRADEDGABAAMECHANISM_CLASS, MYRIAD_CAT(_, CUDAFY_FUN_NAME)),
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)HHGRADEDGABAAMECHANISM_CLASS, 1);
			// Set our device class to the newly-cudafied class object
			((struct MYRIADOBJECT_CLASS*) HHGRADEDGABAAMECHANISM_CLASS)->ONDEVICE_CLASS = 
				(struct MYRIADOBJECT_CLASS*) tmp_mech_c_t;
			
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(HHGRADEDGABAAMECHANISM_CLASS, _dev_t),
					&tmp_mech_c_t,
					sizeof(struct HHGRADEDGABAAMECHANISM_CLASS*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}

	if (!HHGRADEDGABAAMECHANISM_OBJECT)
	{
		HHGRADEDGABAAMECHANISM_OBJECT =
			myriad_new(
				HHGRADEDGABAAMECHANISM_OBJECT,
				MECHANISM_CLASS,
				sizeof(struct HHGRADEDGABAAMECHANISM_OBJECT),
				myriad_ctor, MYRIAD_CAT(HHGRADEDGABAAMECHANISM_OBJECT, MYRIAD_CAT(_, CTOR_FUN_NAME)),
				myriad_cudafy, MYRIAD_CAT(HHGRADEDGABAAMECHANISM_OBJECT, MYRIAD_CAT(_, CUDAFY_FUN_NAME)),
				mechanism_fxn, MYRIAD_CAT(HHGRADEDGABAAMECHANISM_OBJECT, _mech_fun),
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)HHGRADEDGABAAMECHANISM_OBJECT, 1);
			// Set our device class to the newly-cudafied class object
			((struct MYRIADOBJECT_CLASS*) HHGRADEDGABAAMECHANISM_OBJECT)->ONDEVICE_CLASS = 
				(struct MYRIADOBJECT_CLASS*) tmp_mech_t;

			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(HHGRADEDGABAAMECHANISM_OBJECT, _dev_t),
					&tmp_mech_t,
					sizeof(struct HHGRADEDGABAAMECHANISM_OBJECT*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
}




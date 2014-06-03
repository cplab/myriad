#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "myriad_debug.h"

#include "MyriadObject.h"
#include "Mechanism.h"
#include "DCCurrentMech.h"
#include "DCCurrentMech.cuh"

/////////////////////////////////////
// DCCurrentMech Super Overrides //
/////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, DCCURRENTMECHANISM_OBJECT, CTOR_FUN_NAME)
{
	struct DCCURRENTMECHANISM_OBJECT* _self = 
		(struct DCCURRENTMECHANISM_OBJECT*) SUPERCLASS_CTOR(DCCURRENTMECHANISM_OBJECT, self, app);
    
	_self->DCCURRENTMECHANISM_T_START = va_arg(*app, unsigned int);
	_self->DCCURRENTMECHANISM_T_STOP = va_arg(*app, unsigned int);
	_self->DCCURRENTMECHANISM_AMPLITUDE = va_arg(*app, double);
	
	return self;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(MECH_FXN_RET, MECH_FXN_ARGS, DCCURRENTMECHANISM_OBJECT, INDIVIDUAL_MECH_FXN_NAME)
{
	const struct DCCURRENTMECHANISM_OBJECT* self = (const struct DCCURRENTMECHANISM_OBJECT*) _self;

	return (curr_step >= self->DCCURRENTMECHANISM_T_START && curr_step <= self->DCCURRENTMECHANISM_T_STOP) ? self->DCCURRENTMECHANISM_AMPLITUDE : 0.0;
}

//////////////////////////////////////////
// DCCurrentMechClass Super Overrides ////
//////////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, DCCURRENTMECHANISM_CLASS, CUDAFY_FUN_NAME)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct DCCURRENTMECHANISM_CLASS* my_class = (struct DCCURRENTMECHANISM_CLASS*) _self;

		// Make a temporary copy-class because we need to change shit
		struct DCCURRENTMECHANISM_CLASS copy_class = *my_class;
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
                    //TODO: Genericise this out
					(const void*) &DCCurrentMech_mech_fxn_t,
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

const void* DCCURRENTMECHANISM_OBJECT;
const void* DCCURRENTMECHANISM_CLASS;

MYRIAD_FXN_METHOD_HEADER_GEN_NO_SUFFIX(DYNAMIC_INIT_FXN_RET, DYNAMIC_INIT_FXN_ARGS, DCCURRENTMECHANISM_INIT_FXN_NAME)
{
	// initCompartment(init_cuda);
	
	if (!DCCURRENTMECHANISM_CLASS)
	{
		DCCURRENTMECHANISM_CLASS =
			myriad_new(
				MECHANISM_CLASS,
				MECHANISM_CLASS,
				sizeof(struct DCCURRENTMECHANISM_CLASS),
				myriad_cudafy, MYRIAD_CAT(DCCURRENTMECHANISM_CLASS, MYRIAD_CAT(_, CUDAFY_FUN_NAME)),
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)DCCURRENTMECHANISM_CLASS, 1);
			// Set our device class to the newly-cudafied class object
			((struct MYRIADOBJECT_CLASS*) DCCURRENTMECHANISM_CLASS)->ONDEVICE_CLASS = 
				(struct MYRIADOBJECT_CLASS*) tmp_mech_c_t;
			
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(DCCURRENTMECHANISM_CLASS, MYRIAD_CAT(_, DEV_T)),
					&tmp_mech_c_t,
					sizeof(struct DCCURRENTMECHANISM_CLASS*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}

	if (!DCCURRENTMECHANISM_OBJECT)
	{
		DCCURRENTMECHANISM_OBJECT =
			myriad_new(
				DCCURRENTMECHANISM_CLASS,
				MECHANISM_OBJECT,
				sizeof(struct DCCURRENTMECHANISM_OBJECT),
				myriad_ctor, MYRIAD_CAT(DCCURRENTMECHANISM_OBJECT, MYRIAD_CAT(_, CTOR_FUN_NAME)),
				MECH_FXN_NAME_D, MYRIAD_CAT(DCCURRENTMECHANISM_OBJECT, MYRIAD_CAT(_, INDIVIDUAL_MECH_FXN_NAME)),
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)DCCURRENTMECHANISM_OBJECT, 1);
			// Set our device class to the newly-cudafied class object
			((struct MYRIADOBJECT_CLASS*) DCCURRENTMECHANISM_OBJECT)->ONDEVICE_CLASS = 
				(struct MYRIADOBJECT_CLASS*) tmp_mech_t;

			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(DCCURRENTMECHANISM_OBJECT, MYRIAD_CAT(_, DEV_T),
					&tmp_mech_t,
					sizeof(struct DCCURRENTMECHANISM_OBJECT*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
}




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
//static void* DCCurrentMech_ctor(void* _self, va_list* app)
{
	struct DCCURRENTMECHANISM_OBJECT* _self = 
		(struct DCCURRENTMECHANISM_OBJECT*) super_ctor(DCCURRENTMECHANISM_OBJECT, self, app);
    
	_self->DCCURRENTMECHANISM_T_START = va_arg(*app, unsigned int);
	_self->DCCURRENTMECHANISM_T_STOP = va_arg(*app, unsigned int);
	_self->DCCURRENTMECHANISM_AMPLITUDE = va_arg(*app, double);
	
	return self;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(DCCURRENTMECHANISM_MECH_FXN_RET, DCCURRENTMECHANISM_MECH_FXN_ARGS, DCCURRENTMECHANISM_OBJECT, DCCURRENTMECHANISM_MECH_FXN_NAME)
/*
 static double DCCurrentMech_mech_fun(
	void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	)
*/

{
	const struct DCCURRENTMECHANISM_OBJECT* self = (const struct DCCURRENTMECHANISM_OBJECT*) _self;

	return (curr_step >= self->DCCURRENTMECHANISM_T_START && curr_step <= self->DCCURRENTMECHANISM_T_STOP) ? self->DCCURRENTMECHANISM_AMPLITUDE : 0.0;
}

//////////////////////////////////////////
// DCCurrentMechClass Super Overrides ////
//////////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, DCCURRENTMECHANISM_CLASS, CUDAFY_FUN_NAME)
//static void* DCCurrentMechClass_cudafy(void* _self, int clobber)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct DCCURRENTMECHANISM_CLASS* my_class = (struct DCCURRENTMECHANISM_CLASS*) _self;

		// Make a temporary copy-class because we need to change shit
		struct DCCURRENTMECHANISM_CLASS copy_class = *my_class;
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
					(const void*) &MYRIAD_CAT(DCCURRENTMECHANISM_OBJECT, _mech_fxn_t)
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

const void* DCCURRENTMECHANISM_OBJECT;
const void* DCCURRENTMECHANISM_CLASS;

void initDCCurrMech(const int init_cuda)
{
	// initCompartment(init_cuda);
	
	if (!DCCURRENTMECHANISM_CLASS)
	{
		DCCURRENTMECHANISM_CLASS =
			myriad_new(
				MechanismClass,
				MechanismClass,
				sizeof(struct DCCURRENTMECHANISM_CLASS),
				myriad_cudafy, MYRIAD_CAT(DCCURRENTMECHANISM_CLASS, _cudafy),
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)DCCURRENTMECHANISM_CLASS, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) DCCURRENTMECHANISM_CLASS)->device_class = 
				(struct MyriadClass*) tmp_mech_c_t;
			
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(DCCURRENTMECHANISM_CLASS, _dev_t),
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
				Mechanism,
				sizeof(struct DCCURRENTMECHANISM_OBJECT),
				myriad_ctor, MYRIAD_CAT(DCCURRENTMECHANISM_OBJECT, _ctor),
				mechanism_fxn, MYRIAD_CAT(DCCURRENTMECHANISM_OBJECT, _mech_fun),
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)DCCURRENTMECHANISM_OBJECT, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) DCCURRENTMECHANISM_OBJECT)->device_class = 
				(struct MyriadClass*) tmp_mech_t;

			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(DCCURRENTMECHANISM_OBJECT, _dev_t),
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




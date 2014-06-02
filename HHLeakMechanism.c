#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "myriad_debug.h"

#include "MyriadObject.h"
#include "Mechanism.h"
#include "HHSomaCompartment.h"
#include "HHLeakMechanism.h"
#include "HHLeakMechanism.cuh"

#include "HHLeakMechanism_meta.h"

/////////////////////////////////////
// HHLeakMechanism Super Overrides //
/////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, HHLEAKMECHANISM_OBJECT, CTOR_FUN_NAME)
{
	struct HHLEAKMECHANISM_OBJECT* _self = 
		(struct HHLEAKMECHANISM_OBJECT*) SUPERCLASS_CTOR(HHLEAKMECHANISM_OBJECT, self, app);
    
	_self->HHLEAKMECHANISM_G_LEAK = va_arg(*app, double);
	_self->HHLEAKMECHANISM_E_REV = va_arg(*app, double);
	
	return self;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(HHLEAKMECHANISM_MECH_FXN_RET, HHLEAKMECHANISM_MECH_FXN_ARGS, HHLEAKMECHANISM_OBJECT, HHLEAKMECHANISM_MECH_FXN_NAME)
{
	const struct HHLEAKMECHANISM_OBJECT* self = (const struct HHLEAKMECHANISM_OBJECT*) _self;
	const struct HHSOMACOMPARTMENT_OBJECT* c1 = (const struct HHSOMACOMPARTMENT_OBJECT*) pre_comp;
	const struct HHSOMACOMPARTMENT_OBJECT* c2 = (const struct HHSOMACOMPARTMENT_OBJECT*) post_comp;

	//	No extracellular compartment. Current simply "disappears".
	if (c1 == NULL || c1 == c2)
	{
		return -self->HHLEAKMECHANISM_G_LEAK * (c1->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE[curr_step-1] - self->HHLEAKMECHANISM_E_REV);
	}else{
		// @TODO Figure out how to do extracellular compartment calc.
		return 0.0;
	}
}

//////////////////////////////////////////
// HHLeakMechanismClass Super Overrides //
//////////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, HHLEAKMECHANISM_CLASS, CUDAFY_FUN_NAME)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct HHLEAKMECHANISM_CLASS* my_class = (struct HHLEAKMECHANISM_CLASS*) _self;

		// Make a temporary copy-class because we need to change shit
		struct HHLEAKMECHANISM_CLASS copy_class = *my_class;
		struct MYRIADOBJECT_CLASS* copy_class_class = (struct MYRIADOBJECT_CLASS*) &copy_class; // TODO: genericise this when MyriadClass is done
	
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
                    //TOOD: Generecise this out
                    (const void*) &HHLeakMechanism_mech_fxn_t,
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

const void* HHLEAKMECHANISM_OBJECT;
const void* HHLEAKMECHANISM_CLASS;

void initHHLeakMechanism(int init_cuda)
{
	// initCompartment(init_cuda);
	
	if (!HHLEAKMECHANISM_CLASS)
	{
		HHLEAKMECHANISM_CLASS =
			myriad_new(
				MECHANISM_CLASS,
				MECHANISM_CLASS,
				sizeof(struct HHLEAKMECHANISM_CLASS),
				myriad_cudafy, MYRIAD_CAT(HHLEAKMECHANISM_CLASS, MYRIAD_CAT(_, CUDAFY_FUN_NAME)),
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)HHLEAKMECHANISM_CLASS, 1);
			// Set our device class to the newly-cudafied class object
			((struct MYRIADOBJECT_CLASS*) HHLEAKMECHANISM_CLASS)->ONDEVICE_CLASS = 
				(struct MYRIADOBJECT_CLASS*) tmp_mech_c_t;
			
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(HHLEAKMECHANISM_CLASS, _dev_t),
					&tmp_mech_c_t,
					sizeof(struct HHLEAKMECHANISM_CLASS*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}

	if (!HHLEAKMECHANISM_OBJECT)
	{
		HHLEAKMECHANISM_OBJECT =
			myriad_new(
				HHLEAKMECHANISM_CLASS,
				MECHANISM_OBJECT,
				sizeof(struct HHLEAKMECHANISM_OBJECT),
				myriad_ctor, MYRIAD_CAT(HHLEAKMECHANISM_OBJECT, MYRIAD_CAT(_, CTOR_FUN_NAME)),
				mechanism_fxn, MYRIAD_CAT(HHLEAKMECHANISM_OBJECT, _mech_fun),
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)HHLEAKMECHANISM_OBJECT, 1);
			// Set our device class to the newly-cudafied class object
			((struct MYRIADOBJECT_CLASS*) HHLEAKMECHANISM_OBJECT)->ONDEVICE_CLASS = 
				(struct MYRIADOBJECT_CLASS*) tmp_mech_t;

			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(HHLEAKMECHANISM_OBJECT, _dev_t),
					&tmp_mech_t,
					sizeof(struct HHLEAKMECHANISM_OBJECT*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
}




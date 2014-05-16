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
//static void* HHLeakMechanism_ctor(void* _self, va_list* app)
{
	struct HHLEAKMECHANISM_OBJECT* _self = 
		(struct HHLEAKMECHANISM_OBJECT*) super_ctor(HHLEAKMECHANISM_OBJECT, self, app);
    
	_self->HHLEAKMECHANISM_G_LEAK = va_arg(*app, double);
	_self->HHLEAKMECHANISM_E_REV = va_arg(*app, double);
	
	return self;
}

static double HHLeakMechanism_mech_fun(
    void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	)
{
	const struct HHLeakMechanism* self = (const struct HHLeakMechanism*) _self;
	const struct HHSomaCompartment* c1 = (const struct HHSomaCompartment*) pre_comp;
	const struct HHSomaCompartment* c2 = (const struct HHSomaCompartment*) post_comp;

	//	No extracellular compartment. Current simply "disappears".
	if (c1 == NULL || c1 == c2)
	{
		return -self->g_leak * (c1->soma_vm[curr_step-1] - self->e_rev);
	}else{
		// @TODO Figure out how to do extracellular compartment calc.
		return 0.0;
	}
}

//////////////////////////////////////////
// HHLeakMechanismClass Super Overrides //
//////////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, HHLEAKMECHANISM_CLASS, CUDAFY_FUN_NAME)
//static void* HHLeakMechanismClass_cudafy(void* _self, int clobber)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct HHLEAKMECHANISM_CLASS* my_class = (struct HHLEAKMECHANISM_CLASS*) _self;

		// Make a temporary copy-class because we need to change shit
		struct HHLEAKMECHANISM_CLASS copy_class = *my_class;
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
					(const void*) &HHLeakMechanism_mech_fxn_t,
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

const void* HHLEAKMECHANISM_OBJECT;
const void* HHLEAKMECHANISM_CLASS;

void initHHLeakMechanism(int init_cuda)
{
	// initCompartment(init_cuda);
	
	if (!HHLeakMechanismClass)
	{
		HHLeakMechanismClass =
			myriad_new(
				MechanismClass,
				MechanismClass,
				sizeof(struct HHLeakMechanismClass),
				myriad_cudafy, HHLeakMechanismClass_cudafy,
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)HHLeakMechanismClass, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) HHLeakMechanismClass)->device_class = 
				(struct MyriadClass*) tmp_mech_c_t;
			
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &HHLeakMechanismClass_dev_t,
					&tmp_mech_c_t,
					sizeof(struct HHLeakMechanismClass*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}

	if (!HHLeakMechanism)
	{
		HHLeakMechanism =
			myriad_new(
				HHLeakMechanismClass,
				Mechanism,
				sizeof(struct HHLeakMechanism),
				myriad_ctor, HHLeakMechanism_ctor,
				mechanism_fxn, HHLeakMechanism_mech_fun,
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)HHLeakMechanism, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) HHLeakMechanism)->device_class = 
				(struct MyriadClass*) tmp_mech_t;

			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &HHLeakMechanism_dev_t,
					&tmp_mech_t,
					sizeof(struct HHLeakMechanism*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
}




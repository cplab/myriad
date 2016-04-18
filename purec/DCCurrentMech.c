#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "MyriadObject.h"
#include "Mechanism.h"
#include "DCCurrentMech.h"
#include "DCCurrentMech.cuh"

/////////////////////////////////////
// DCCurrentMech Super Overrides //
/////////////////////////////////////

static void* DCCurrentMech_ctor(void* _self, va_list* app)
{
	struct DCCurrentMech* self = 
		(struct DCCurrentMech*) super_ctor(DCCurrentMech, _self, app);
    
	self->t_start = va_arg(*app, uint_fast32_t);
	self->t_stop = va_arg(*app, uint_fast32_t);
    self->amplitude = va_arg(*app, double);
	
	return self;
}

static scalar DCCurrentMech_mech_fun(void* _self,
                                     void* pre_comp,
                                     void* post_comp,
                                     const scalar global_time,
                                     const uint_fast32_t curr_step)
{
	const struct DCCurrentMech* self = (const struct DCCurrentMech*) _self;

	return (curr_step >= self->t_start && curr_step <= self->t_stop) ? self->amplitude : 0.0;
}

//////////////////////////////////////////
// DCCurrentMechClass Super Overrides //
//////////////////////////////////////////

static void* DCCurrentMechClass_cudafy(void* _self, int clobber)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct DCCurrentMechClass* my_class = (struct DCCurrentMechClass*) _self;

		// Make a temporary copy-class because we need to change shit
		struct DCCurrentMechClass copy_class = *my_class;
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
					(const void*) &DCCurrentMech_mech_fxn_t,
					sizeof(void*),
					0,
					cudaMemcpyDeviceToHost
					)
				);
			copy_class._.m_mech_fxn = my_mech_fun;
		
			// DEBUG_PRINTF("Copy Class mech fxn: %p\n", (void*) my_mech_fun);
		
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

const void* DCCurrentMech;
const void* DCCurrentMechClass;

void initDCCurrMech(void)
{
	if (!DCCurrentMechClass)
	{
		DCCurrentMechClass =
			myriad_new(
				MechanismClass,
				MechanismClass,
				sizeof(struct DCCurrentMechClass),
				myriad_cudafy, DCCurrentMechClass_cudafy,
				0
			);
		
#ifdef CUDA
        void* tmp_mech_c_t = myriad_cudafy((void*)DCCurrentMechClass, 1);
        // Set our device class to the newly-cudafied class object
        ((struct MyriadClass*) DCCurrentMechClass)->device_class = 
            (struct MyriadClass*) tmp_mech_c_t;
			
        CUDA_CHECK_RETURN(
            cudaMemcpyToSymbol(
                (const void*) &DCCurrentMechClass_dev_t,
                &tmp_mech_c_t,
                sizeof(struct DCCurrentMechClass*),
                0,
                cudaMemcpyHostToDevice
                )
            );
#endif
	}

	if (!DCCurrentMech)
	{
		DCCurrentMech =
			myriad_new(
				DCCurrentMechClass,
				Mechanism,
				sizeof(struct DCCurrentMech),
				myriad_ctor, DCCurrentMech_ctor,
				mechanism_fxn, DCCurrentMech_mech_fun,
				0
			);
		
#ifdef CUDA
        void* tmp_mech_t = myriad_cudafy((void*)DCCurrentMech, 1);
        // Set our device class to the newly-cudafied class object
        ((struct MyriadClass*) DCCurrentMech)->device_class = 
            (struct MyriadClass*) tmp_mech_t;

        CUDA_CHECK_RETURN(
            cudaMemcpyToSymbol(
                (const void*) &DCCurrentMech_dev_t,
                &tmp_mech_t,
                sizeof(struct DCCurrentMech*),
                0,
                cudaMemcpyHostToDevice
                )
            );
#endif
	}
}




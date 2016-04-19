#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>

#include "MyriadObject.h"
#include "Mechanism.h"
#include "Mechanism.cuh"

///////////////////////////////
// Mechanism Super Overrides //
///////////////////////////////

// for some reason, the above macro is not concatenating correctly
static void* Mechanism_ctor(void* _self, va_list* app)
{
	struct Mechanism* self = (struct Mechanism*) super_ctor(Mechanism, _self, app);
	
	self->source_id = va_arg(*app, uint_fast32_t);

	return _self;
}

//////////////////////////
// Native Mechanism Fxn //
//////////////////////////

static double Mechanism_mechanism_fxn(void* _self,
                                      void* pre_comp,
                                      void* post_comp,
                                      const double global_time,
                                      const uint_fast32_t curr_step)
{
	return 0.0;
}

////////////////////////////////////
// MechanismClass Super Overrides //
////////////////////////////////////


static void* MechanismClass_ctor(void* _self, va_list* app)
{
	struct MechanismClass* self = (struct MechanismClass*) super_ctor(MechanismClass, _self, app);

    enum MyriadMethods selector = 0; selector = va_arg(app, enum MyriadMethods);

	while (selector)
	{
		const voidf method = va_arg(*app, voidf);
		
		if (selector == MECH_SIMUL)
		{
			*(voidf *) &self->m_mech_fxn = method;
		}

		selector = va_arg(*app, enum MyriadMethods);
	}

	return self;
}

static void* MechanismClass_cudafy(void* _self, int clobber)
{
#ifdef CUDA
    // We know that we're actually a mechanism class
    struct MechanismClass* my_class = (struct MechanismClass*) _self;

    // Make a temporary copy-class because we need to change shit
    struct MechanismClass copy_class = *my_class; // Assignment to stack avoids calloc/memcpy
    struct MyriadClass* copy_class_class = (struct MyriadClass*) &copy_class;

    // TODO: Find a better way to get function pointers for on-card functions
    mech_fun_t my_mech_fun = NULL;
    CUDA_CHECK_RETURN(
        cudaMemcpyFromSymbol(
            (void**) &my_mech_fun,
            (const void*) &Mechanism_cuda_mechanism_fxn_t,
            sizeof(void*),
            0,
            cudaMemcpyDeviceToHost
            )
        );
    copy_class.m_mech_fxn = my_mech_fun;
    // DEBUG_PRINTF("Copy Class mech fxn: %p\n", (void*) my_mech_fun);
	
    // !!!!!!!!! IMPORTANT !!!!!!!!!!!!!!
    // By default we clobber the copy_class_class' superclass with
    // the superclass' device_class' on-GPU address value. 
    // To avoid cloberring this value (e.g. if an underclass has already
    // clobbered it), the clobber flag should be 0.
    if (clobber)
    {
        const struct MyriadClass* super_class = (const struct MyriadClass*) MyriadClass;
        memcpy((void**) &copy_class_class->super, &super_class->device_class, sizeof(void*));
    }

    // This works because super methods rely on the given class'
    // semi-static superclass definition, not it's ->super attribute.
    return super_cudafy(MechanismClass, (void*) &copy_class, 0);
#else
    return NULL;
#endif
}

/////////////////////////////////////
// Reference Object Initialization //
/////////////////////////////////////

const void *MechanismClass, *Mechanism;

void initMechanism(void)
{
	if (!MechanismClass)
	{
		MechanismClass = 
			myriad_new(
				   MyriadClass,
				   MyriadClass,
				   sizeof(struct MechanismClass),
				   CTOR, MechanismClass_ctor,
				   CUDAFY, MechanismClass_cudafy,
				   0
			);
		struct MyriadObject* mech_class_obj = NULL; mech_class_obj = (struct MyriadObject*) MechanismClass;
		memcpy( (void**) &mech_class_obj->m_class, &MechanismClass, sizeof(void*));

#ifdef CUDA
        void* tmp_mech_c_t = myriad_cudafy((void*)MechanismClass, 1);
        ((struct MyriadClass*) MechanismClass)->device_class = (struct MyriadClass*) tmp_mech_c_t;
        CUDA_CHECK_RETURN(
            cudaMemcpyToSymbol(
                (const void*) &MechanismClass_dev_t,
                &tmp_mech_c_t,
                sizeof(struct MechanismClass*),
                0,
                cudaMemcpyHostToDevice
                )
            );
#endif
    }
    
	if (!Mechanism)
	{
		Mechanism = 
			myriad_new(
				   MechanismClass,
				   MyriadObject,
				   sizeof(struct Mechanism),
				   CTOR, Mechanism_ctor,
				   MECH_SIMUL, Mechanism_mechanism_fxn,
				   0
			);

#ifdef CUDA
			void* tmp_mech_t = myriad_cudafy((void*)Mechanism, 1);
			((struct MyriadClass*) Mechanism)->device_class = (struct MyriadClass*) tmp_mech_t;
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &Mechanism_dev_t,
					&tmp_mech_t,
					sizeof(struct Mechanism*),
					0,
					cudaMemcpyHostToDevice
					)
				);

#endif
    }
}

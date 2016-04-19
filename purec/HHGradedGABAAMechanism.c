#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "MyriadObject.h"
#include "Mechanism.h"
#include "HHSomaCompartment.h"
#include "HHGradedGABAAMechanism.h"
#include "HHGradedGABAAMechanism.cuh"

///////////////////////////////////////
// HHGradedGABAAMechanism Super Overrides //
///////////////////////////////////////

static void* HHGradedGABAAMechanism_ctor(void* _self, va_list* app)
{
	struct HHGradedGABAAMechanism* self = 
		(struct HHGradedGABAAMechanism*) super_ctor(HHGradedGABAAMechanism, _self, app);

    double* pot_g_s = va_arg(*app, double*);
	const double g_s_init = va_arg(*app, double);
	if (pot_g_s != NULL)
    {
        memcpy(self->g_s, pot_g_s, SIMUL_LEN * sizeof(double));
        self->g_s[0] = g_s_init;
    }

	self->g_max = va_arg(*app, double);
	self->theta = va_arg(*app, double);
	self->sigma = va_arg(*app, double);
	self->tau_alpha = va_arg(*app, double);
	self->tau_beta = va_arg(*app, double);
	self->gaba_rev = va_arg(*app, double);

	return self;
}

static double HHGradedGABAAMechanism_mech_fun(void* _self,
                                              void* pre_comp,
                                              void* post_comp,
                                              const double dt,
                                              const double global_time,
                                              const uint_fast32_t curr_step)
{
	struct HHGradedGABAAMechanism* self = (struct HHGradedGABAAMechanism*) _self;
	const struct HHSomaCompartment* c1 = (const struct HHSomaCompartment*) pre_comp;
	const struct HHSomaCompartment* c2 = (const struct HHSomaCompartment*) post_comp;

	//	Channel dynamics calculation
	const double pre_vm = c1->vm[curr_step-1];
	const double post_vm = c2->vm[curr_step-1];
	const double prev_g_s = self->g_s[curr_step-1];

	const double fv = 1.0 / (1.0 + exp((pre_vm - self->theta)/-self->sigma));
	self->g_s[curr_step] += dt * (self->tau_alpha * fv * (1.0 - prev_g_s) - self->tau_beta * prev_g_s);

	const double I_GABA = -self->g_max * prev_g_s * (post_vm - self->gaba_rev);
	return I_GABA;
}

////////////////////////////////////////////
// HHGradedGABAAMechanismClass Super Overrides //
////////////////////////////////////////////

static void* HHGradedGABAAMechanismClass_cudafy(void* _self, int clobber)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct HHGradedGABAAMechanismClass* my_class = (struct HHGradedGABAAMechanismClass*) _self;

		// Make a temporary copy-class because we need to change shit
		struct HHGradedGABAAMechanismClass copy_class = *my_class;
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
					(const void*) &HHGradedGABAAMechanism_mech_fxn_t,
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

const void* HHGradedGABAAMechanism;
const void* HHGradedGABAAMechanismClass;

void initHHGradedGABAAMechanism(const bool init_cuda)
{
	if (!HHGradedGABAAMechanismClass)
	{
		HHGradedGABAAMechanismClass =
			myriad_new(
				MechanismClass,
				MechanismClass,
				sizeof(struct HHGradedGABAAMechanismClass),
				CUDAFY, HHGradedGABAAMechanismClass_cudafy,
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)HHGradedGABAAMechanismClass, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) HHGradedGABAAMechanismClass)->device_class = 
				(struct MyriadClass*) tmp_mech_c_t;
			
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &HHGradedGABAAMechanismClass_dev_t,
					&tmp_mech_c_t,
					sizeof(struct HHGradedGABAAMechanismClass*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}

	if (!HHGradedGABAAMechanism)
	{
		HHGradedGABAAMechanism =
			myriad_new(
				HHGradedGABAAMechanismClass,
				Mechanism,
				sizeof(struct HHGradedGABAAMechanism),
				CTOR, HHGradedGABAAMechanism_ctor,
				MECH_SIMUL, HHGradedGABAAMechanism_mech_fun,
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)HHGradedGABAAMechanism, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) HHGradedGABAAMechanism)->device_class = 
				(struct MyriadClass*) tmp_mech_t;

			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &HHGradedGABAAMechanism_dev_t,
					&tmp_mech_t,
					sizeof(struct HHGradedGABAAMechanism*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
}




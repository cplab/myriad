#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "MyriadObject.h"
#include "Mechanism.h"
#include "HHSomaCompartment.h"
#include "HHSpikeGABAAMechanism.h"

///////////////////////////////////////
// HHSpikeGABAAMechanism Super Overrides //
///////////////////////////////////////

static void* HHSpikeGABAAMechanism_ctor(void* _self, va_list* app)
{
	struct HHSpikeGABAAMechanism* self = 
		(struct HHSpikeGABAAMechanism*) super_ctor(HHSpikeGABAAMechanism, _self, app);

    self->prev_vm_thresh = va_arg(*app, double); 
    self->t_fired = va_arg(*app, double);
	self->g_max = va_arg(*app, double);
	self->tau_alpha = va_arg(*app, double);
	self->tau_beta = va_arg(*app, double);
	self->gaba_rev = va_arg(*app, double);

    // Automatic calculations for t_p and A-bar, from Guoshi
    self->peak_cond_t = ((self->tau_alpha * self->tau_beta) /
        (self->tau_beta - self->tau_alpha)) * 
        log(self->tau_beta / self->tau_alpha);

    self->norm_const = 1.0 / 
        (exp(-self->peak_cond_t/self->tau_beta) - 
         exp(-self->peak_cond_t/self->tau_alpha));

	return self;
}

static double HHSpikeGABAAMechanism_mech_fun(void* _self,
                                             void* pre_comp,
                                             void* post_comp,
                                             const double dt,
                                             const double global_time,
                                             const uint64_t curr_step)
{
	struct HHSpikeGABAAMechanism* self = (struct HHSpikeGABAAMechanism*) _self;
	const struct HHSomaCompartment* c1 = (const struct HHSomaCompartment*) pre_comp;
	const struct HHSomaCompartment* c2 = (const struct HHSomaCompartment*) post_comp;

	//	Channel dynamics calculation
    const double pre_pre_vm = (curr_step > 1) ? c1->vm[curr_step-2] : INFINITY;
	const double pre_vm = c1->vm[curr_step-1];
	const double post_vm = c2->vm[curr_step-1];
    
    // If we just fired
    if (pre_vm > self->prev_vm_thresh && pre_pre_vm < self->prev_vm_thresh)
    {
        self->t_fired = global_time;
    }

    if (self->t_fired != -INFINITY)
    {
        const double g_s = exp(-(global_time - self->t_fired) / self->tau_beta) - 
            exp(-(global_time - self->t_fired) / self->tau_alpha);
        const double I_GABA = self->norm_const * -self->g_max * g_s * (post_vm - self->gaba_rev);
        return I_GABA;        
    } else {
        return 0.0;
    }
}

////////////////////////////////////////////
// HHSpikeGABAAMechanismClass Super Overrides //
////////////////////////////////////////////

static void* HHSpikeGABAAMechanismClass_cudafy(void* _self, int clobber)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct HHSpikeGABAAMechanismClass* my_class = (struct HHSpikeGABAAMechanismClass*) _self;

		// Make a temporary copy-class because we need to change shit
		struct HHSpikeGABAAMechanismClass copy_class = *my_class;
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
					(const void*) &HHSpikeGABAAMechanism_mech_fxn_t,
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

const void* HHSpikeGABAAMechanism;
const void* HHSpikeGABAAMechanismClass;

void initHHSpikeGABAAMechanism(const bool init_cuda)
{
	if (!HHSpikeGABAAMechanismClass)
	{
		HHSpikeGABAAMechanismClass =
			myriad_new(
				MechanismClass,
				MechanismClass,
				sizeof(struct HHSpikeGABAAMechanismClass),
				myriad_cudafy, HHSpikeGABAAMechanismClass_cudafy,
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)HHSpikeGABAAMechanismClass, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) HHSpikeGABAAMechanismClass)->device_class = 
				(struct MyriadClass*) tmp_mech_c_t;
			
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &HHSpikeGABAAMechanismClass_dev_t,
					&tmp_mech_c_t,
					sizeof(struct HHSpikeGABAAMechanismClass*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}

	if (!HHSpikeGABAAMechanism)
	{
		HHSpikeGABAAMechanism =
			myriad_new(
				HHSpikeGABAAMechanismClass,
				Mechanism,
				sizeof(struct HHSpikeGABAAMechanism),
				myriad_ctor, HHSpikeGABAAMechanism_ctor,
				mechanism_fxn, HHSpikeGABAAMechanism_mech_fun,
				0
			);
		
		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)HHSpikeGABAAMechanism, 1);
			// Set our device class to the newly-cudafied class object
			((struct MyriadClass*) HHSpikeGABAAMechanism)->device_class = 
				(struct MyriadClass*) tmp_mech_t;

			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &HHSpikeGABAAMechanism_dev_t,
					&tmp_mech_t,
					sizeof(struct HHSpikeGABAAMechanism*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
}




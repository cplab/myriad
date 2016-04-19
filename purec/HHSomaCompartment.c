#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

#include "MyriadObject.h"
#include "HHSomaCompartment.h"
#include "HHSomaCompartment.cuh"

///////////////////////////////////////
// HHSomaCompartment Super Overrides //
///////////////////////////////////////
static void* HHSomaCompartment_ctor(void* _self, va_list* app)
{
    struct HHSomaCompartment* self =
        (struct HHSomaCompartment*) super_ctor(HHSomaCompartment, _self, app);

    const double* restrict vm = va_arg(*app, double*);
    const double init_vm = va_arg(*app, double);
    self->cm = va_arg(*app, double);

    // If the given vm is non-NULL, we assume it contains data and copy it.
    if (vm != NULL)
    {
        memcpy(self->vm, vm, SIMUL_LEN * sizeof(double));
        self->vm[0] = init_vm;
    }

    return _self;
}

static void* HHSomaCompartment_cudafy(void* _self, int clobber)
{
#ifdef CUDA
    return super_cudafy(HHSomaCompartment, _self, 0);
#else
    return NULL;
#endif
}

static void HHSomaCompartment_decudafy(void* _self, void* cuda_self)
{
#ifdef CUDA
    super_decudafy(Compartment, _self, cuda_self);
#endif
	return;
}

static int HHSomaCompartment_dtor(void* _self)
{
	struct HHSomaCompartment* self = (struct HHSomaCompartment*) _self;
	return super_dtor(Compartment, self);
}

static void HHSomaCompartment_simul_fxn(void* _self,
                                        void** network,
                                        const double global_time,
                                        const uint_fast32_t curr_step)
{
	struct HHSomaCompartment* self = (struct HHSomaCompartment*) _self;

	double I_sum = 0.0;

	//	Calculate mechanism contribution to current term
	for (uint_fast32_t i = 0; i < self->_.num_mechs; i++)
	{
		struct Mechanism* curr_mech = self->_.my_mechs[i]; // TODO: GENERICSE DIS
		struct Compartment* pre_comp = network[curr_mech->source_id];

		//TODO: Make this conditional on specific Mechanism types
		//if (curr_mech->fx_type == CURRENT_FXN)
		I_sum += mechanism_fxn(curr_mech, pre_comp, self, global_time, curr_step);
	}

	//	Calculate new membrane voltage: (dVm) + prev_vm
	self->vm[curr_step] = (DT * (I_sum) / (self->cm)) + self->vm[curr_step - 1];

	return;
}

////////////////////////////////////////////
// HHSomaCompartmentClass Super Overrides //
////////////////////////////////////////////

static void* HHSomaCompartmentClass_cudafy(void* _self, int clobber)
{
#ifdef CUDA
    // We know what class we are
    struct HHSomaCompartmentClass* my_class = (struct HHSomaCompartmentClass*) _self;

    // Make a temporary copy-class because we need to change shit
    struct HHSomaCompartmentClass copy_class = *my_class;
    struct MyriadClass* copy_class_class = (struct MyriadClass*) &copy_class;
	
    // !!!!!!!!! IMPORTANT !!!!!!!!!!!!!!
    // By default we clobber the copy_class_class' superclass with
    // the superclass' device_class' on-GPU address value. 
    // To avoid cloberring this value (e.g. if an underclass has already
    // clobbered it), the clobber flag should be 0.
    if (clobber)
    {
        // TODO: Find a better way to get function pointers for on-card functions
        compartment_simul_fxn_t my_comp_fun = NULL;
        CUDA_CHECK_RETURN(
            cudaMemcpyFromSymbol(
                (void**) &my_comp_fun,
                (const void*) &HHSomaCompartment_simul_fxn_t,
                sizeof(void*),
                0,
                cudaMemcpyDeviceToHost
                )
            );
        copy_class._.m_compartment_simul_fxn = my_comp_fun;
		
        // DEBUG_PRINTF("Copy Class comp fxn: %p\n", (void*) my_comp_fun);
		
        const struct MyriadClass* super_class = (const struct MyriadClass*) CompartmentClass;
        memcpy((void**) &copy_class_class->super, &super_class->device_class, sizeof(void*));
    }

    // This works because super methods rely on the given class'
    // semi-static superclass definition, not it's ->super attribute.
    // Note that we don't want to clobber, so we set it to 0.
    return super_cudafy(CompartmentClass, (void*) &copy_class, 0);
#else
    // Can't cudafy if there's no CUDA
    return NULL;
#endif
}

////////////////////////////
// Dynamic Initialization //
////////////////////////////

const void *HHSomaCompartment, *HHSomaCompartmentClass;

void initHHSomaCompartment(void)
{
	if (!HHSomaCompartmentClass)
	{
		HHSomaCompartmentClass =
			myriad_new(
				CompartmentClass,
				CompartmentClass,
				sizeof(struct HHSomaCompartmentClass),
				CUDAFY, HHSomaCompartmentClass_cudafy,
				0
			);

#ifdef CUDA
        void* tmp_comp_c_t = myriad_cudafy((void*)HHSomaCompartmentClass, 1);
        ((struct MyriadClass*) HHSomaCompartmentClass)->device_class = (struct MyriadClass*) tmp_comp_c_t;
        CUDA_CHECK_RETURN(
            cudaMemcpyToSymbol(
                (const void*) &HHSomaCompartmentClass_dev_t,
                &tmp_comp_c_t,
                sizeof(struct HHSomaCompartmentClass*),
                0,
                cudaMemcpyHostToDevice
                )
            );
#endif
	}

	if (!HHSomaCompartment)
	{
		HHSomaCompartment =
			myriad_new(
				HHSomaCompartmentClass,
				Compartment,
				sizeof(struct HHSomaCompartment),
				CTOR, HHSomaCompartment_ctor,
				DTOR, HHSomaCompartment_dtor,
				CUDAFY, HHSomaCompartment_cudafy,
				DECUDAFY, HHSomaCompartment_decudafy,
				COMPARTMENT_SIMUL, HHSomaCompartment_simul_fxn,
				0
			);

#ifdef CUDA
        void* tmp_mech_t = myriad_cudafy((void*)HHSomaCompartment, 1);
        ((struct MyriadClass*) HHSomaCompartment)->device_class = (struct MyriadClass*) tmp_mech_t;
        CUDA_CHECK_RETURN(
            cudaMemcpyToSymbol(
                (const void*) &HHSomaCompartment_dev_t,
                &tmp_mech_t,
                sizeof(struct HHSomaCompartment*),
                0,
                cudaMemcpyHostToDevice
                )
            );
#endif
	}
}

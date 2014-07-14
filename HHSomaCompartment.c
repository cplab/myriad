#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "myriad_debug.h"

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

    self->vm_len = va_arg(*app, unsigned int);
    self->vm = va_arg(*app, double*);
    const double init_vm = va_arg(*app, double);
    self->cm = va_arg(*app, double);

    // If the given length is non-zero but the pointer is NULL,
    // we do the allocation ourselves.
    if (self->vm == NULL && self->vm_len > 0)
    {
        self->vm = (double*) calloc(self->vm_len, sizeof(double));
        assert(self->vm && "Failed to allocate soma membrane voltage array");
        self->vm[0] = init_vm;
    }

    return _self;
}

static void* HHSomaCompartment_cudafy(void* _self, int clobber)
{
	#ifdef CUDA
	{
		const size_t my_size = myriad_size_of(_self);
		struct HHSomaCompartment* self = (struct HHSomaCompartment*) _self;
		struct HHSomaCompartment* self_copy = (struct HHSomaCompartment*) calloc(1, my_size);
		
		memcpy(self_copy, HHSomaCompartment, my_size);

		double* tmp_alias = NULL;
		
		// Make mirror on-GPU array 
		CUDA_CHECK_RETURN(
			cudaMalloc(
				(void**) &tmp_alias,
				self_copy->vm_len * sizeof(double)
				)
			);

		// Copy contents over to GPU
		CUDA_CHECK_RETURN(
			cudaMemcpy(
				(void*) tmp_alias,
				(void*) self->vm,
				self_copy->vm_len * sizeof(double),
				cudaMemcpyHostToDevice
				)
			);

		self_copy->vm = tmp_alias;

		return super_cudafy(HHSomaCompartment, self_copy, 0);
	}
	#else
	{
	    return NULL;
    }
	#endif
}

static void HHSomaCompartment_decudafy(void* _self, void* cuda_self)
{
	#ifdef CUDA
	{
		struct HHSomaCompartment* self = (struct HHSomaCompartment*) _self;

		double* from_gpu_soma = NULL;
		CUDA_CHECK_RETURN(
			cudaMemcpy(
				(void*) &from_gpu_soma,
				(void*) cuda_self + offsetof(struct HHSomaCompartment, vm),
				sizeof(double*),
				cudaMemcpyDeviceToHost
				)
			);

		CUDA_CHECK_RETURN(
			cudaMemcpy(
				(void*) self->vm,
				(void*) from_gpu_soma,
				self->vm_len * sizeof(double),
				cudaMemcpyDeviceToHost
				)
			);

		super_decudafy(Compartment, self, cuda_self);
	}
	#endif

	return;
}

static int HHSomaCompartment_dtor(void* _self)
{
	struct HHSomaCompartment* self = (struct HHSomaCompartment*) _self;

	free(self->vm);

	return super_dtor(Compartment, self);
}

static void HHSomaCompartment_simul_fxn(
	void* _self,
	void** network,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	)
{
	struct HHSomaCompartment* self = (struct HHSomaCompartment*) _self;

	double I_sum = 0.0;

	//	Calculate mechanism contribution to current term
	for (unsigned int i = 0; i < self->_.num_mechs; i++)
	{
		struct Mechanism* curr_mech = self->_.my_mechs[i]; // TODO: GENERICSE DIS
		struct Compartment* pre_comp = network[curr_mech->source_id];

		//TODO: Make this conditional on specific Mechanism types
		//if (curr_mech->fx_type == CURRENT_FXN)
		I_sum += mechanism_fxn(curr_mech, pre_comp, self, dt, global_time, curr_step);
	}

	//	Calculate new membrane voltage: (dVm) + prev_vm
	self->vm[curr_step] = (dt * (I_sum) / (self->cm)) + self->vm[curr_step - 1];

	return;
}

////////////////////////////////////////////
// HHSomaCompartmentClass Super Overrides //
////////////////////////////////////////////

static void* HHSomaCompartmentClass_cudafy(void* _self, int clobber)
{
	#ifdef CUDA
	{
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
		
			DEBUG_PRINTF("Copy Class comp fxn: %p\n", my_comp_fun);
		
			const struct MyriadClass* super_class = (const struct MyriadClass*) CompartmentClass;
			memcpy((void**) &copy_class_class->super, &super_class->device_class, sizeof(void*));
		}

		// This works because super methods rely on the given class'
		// semi-static superclass definition, not it's ->super attribute.
		// Note that we don't want to clobber, so we set it to 0.
		return super_cudafy(CompartmentClass, (void*) &copy_class, 0);
	}
	#else
	{
	    // Can't cudafy if there's no CUDA
    	return NULL;
    }
	#endif
}

////////////////////////////
// Dynamic Initialization //
////////////////////////////

const void* HHSomaCompartment;
const void* HHSomaCompartmentClass;

void initHHSomaCompartment(int init_cuda)
{
	// initCompartment(init_cuda);

	if (!HHSomaCompartmentClass)
	{
		HHSomaCompartmentClass =
			myriad_new(
				CompartmentClass,
				CompartmentClass,
				sizeof(struct HHSomaCompartmentClass),
				myriad_cudafy, HHSomaCompartmentClass_cudafy,
				0
			);

		#ifdef CUDA
		if (init_cuda)
		{
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
		}
		#endif
	}

	if (!HHSomaCompartment)
	{
		HHSomaCompartment =
			myriad_new(
				HHSomaCompartmentClass,
				Compartment,
				sizeof(struct HHSomaCompartment),
				myriad_ctor, HHSomaCompartment_ctor,
				myriad_dtor, HHSomaCompartment_dtor,
				myriad_cudafy, HHSomaCompartment_cudafy,
				myriad_decudafy, HHSomaCompartment_decudafy,
				simul_fxn, HHSomaCompartment_simul_fxn,
				0
			);

		#ifdef CUDA
		if (init_cuda)
		{
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
		}
		#endif
	}
}

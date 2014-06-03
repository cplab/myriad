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

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, HHSOMACOMPARTMENT_OBJECT, CTOR_FUN_NAME)
{
	struct HHSOMACOMPARTMENT_OBJECT* _self = 
		(struct HHSOMACOMPARTMENT_OBJECT*) SUPERCLASS_CTOR(HHSOMACOMPARTMENT_OBJECT, self, app);
	
	_self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE_LENGTH = va_arg(*app, unsigned int);
	_self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE = va_arg(*app, double*);
	const double init_vm = va_arg(*app, double);
	_self->HHSOMACOMPARTMENT_CAPACITANCE = va_arg(*app, double);

	// If the given length is non-zero but the pointer is NULL,
	// we do the allocation ourselves.
	if (_self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE == NULL && _self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE_LENGTH > 0)
	{
		_self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE = (double*) calloc(_self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE_LENGTH, sizeof(double));
		assert(_self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE && "Failed to allocate soma membrane voltage array");
		_self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE[0] = init_vm;
	}

	return _self;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, HHSOMACOMPARTMENT_OBJECT, CUDAFY_FUN_NAME)
{
	#ifdef CUDA
	{
		const size_t my_size = myriad_size_of(_self);
		struct HHSOMACOMPARTMENT_OBJECT* self = (struct HHSOMACOMPARTMENT_OBJECT*) _self;
		struct HHSOMACOMPARTMENT_OBJECT* self_copy = (struct HHSOMACOMPARTMENT_OBJECT*) calloc(1, my_size);
		
		memcpy(self_copy, HHSOMACOMPARTMENT_OBJECT, my_size);

		double* tmp_alias = NULL;
		
		// Make mirror on-GPU array 
		CUDA_CHECK_RETURN(
			cudaMalloc(
				(void**) &tmp_alias,
				self_copy->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE_LENGTH * sizeof(double)
				)
			);

		// Copy contents over to GPU
		CUDA_CHECK_RETURN(
			cudaMemcpy(
				(void*) tmp_alias,
				(void*) self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE,
				self_copy->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE_LENGTH * sizeof(double),
				cudaMemcpyHostToDevice
				)
			);

		self_copy->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE = tmp_alias;

		return SUPERCLASS_CUDAFY(HHSOMACOMPARTMENT_OBJECT, self_copy, 0);
	}
	#else
	{
	    return NULL;
    }
	#endif
}

static MYRIAD_FXN_METHOD_HEADER_GEN(DECUDAFY_FUN_RET, DECUDAFY_FUN_ARGS, HHSOMACOMPARTMENT_OBJECT, DECUDAFY_FUN_NAME)
{
	#ifdef CUDA
	{
		struct HHSOMACOMPARTMENT_OBJECT* self = (struct HHSOMACOMPARTMENT_OBJECT*) _self;

		double* from_gpu_soma = NULL;
		CUDA_CHECK_RETURN(
			cudaMemcpy(
				(void*) &from_gpu_soma,
				(void*) cuda_self + offsetof(struct HHSOMACOMPARTMENT_OBJECT, HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE),
				sizeof(double*),
				cudaMemcpyDeviceToHost
				)
			);

		CUDA_CHECK_RETURN(
			cudaMemcpy(
				(void*) self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE,
				(void*) from_gpu_soma,
				self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE_LENGTH * sizeof(double),
				cudaMemcpyDeviceToHost
				)
			);

		SUPERCLASS_DECUDAFY(COMPARTMENT_OBJECT, self, cuda_self);
	}
	#endif

	return;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(DTOR_FUN_RET, DTOR_FUN_ARGS, HHSOMACOMPARTMENT_OBJECT, DTOR_FUN_NAME)
{
	struct HHSOMACOMPARTMENT_OBJECT* _self = (struct HHSOMACOMPARTMENT_OBJECT*) self;

	free(_self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE);

	return SUPERCLASS_DTOR(COMPARTMENT_OBJECT, self);
}

static MYRIAD_FXN_METHOD_HEADER_GEN(SIMUL_FXN_RET, SIMUL_FXN_ARGS, HHSOMACOMPARTMENT_OBJECT, SIMUL_FXN_NAME_D)
{
	struct HHSOMACOMPARTMENT_OBJECT* _self = (struct HHSOMACOMPARTMENT_OBJECT*) self;

	double I_sum = 0.0;

	//	Calculate mechanism contribution to current term
	for (unsigned int i = 0; i < _self->_.NUM_MECHS; i++)
	{
		struct MECHANISM_OBJECT* curr_mech = _self->_.MY_MECHS[i];
		struct COMPARTMENT_OBJECT* pre_comp = network[curr_mech->COMPARTMENT_PREMECH_SOURCE_ID];

		//TODO: Make this conditional on specific Mechanism types
		//if (curr_mech->fx_type == CURRENT_FXN)
		I_sum += MECH_FXN_NAME_D(curr_mech, pre_comp, self, dt, global_time, curr_step);
	}

	//	Calculate new membrane voltage: (dVm) + prev_vm
	_self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE[curr_step] = (dt * (I_sum) / (_self->HHSOMACOMPARTMENT_CAPACITANCE)) + _self->HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE[curr_step - 1];

	return;
}

////////////////////////////////////////////
// HHSomaCompartmentClass Super Overrides //
////////////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, HHSOMACOMPARTMENT_CLASS, CUDAFY_FUN_NAME)
{
	#ifdef CUDA
	{
		// We know what class we are
		struct HHSOMACOMPARTMENT_CLASS* my_class = (struct HHSOMACOMPARTMENT_CLASS*) _self;

		// Make a temporary copy-class because we need to change shit
		struct HHSOMACOMPARTMENT_CLASS copy_class = *my_class;
		struct MYRIADOBJECT_CLASS* copy_class_class = (struct MYRIADOBJECT_CLASS*) &copy_class;
	
		// !!!!!!!!! IMPORTANT !!!!!!!!!!!!!!
		// By default we clobber the copy_class_class' superclass with
		// the superclass' device_class' on-GPU address value. 
		// To avoid cloberring this value (e.g. if an underclass has already
		// clobbered it), the clobber flag should be 0.
		if (clobber)
		{
			// TODO: Find a better way to get function pointers for on-card functions
			SIMUL_FXN_TYPEDEF_NAME my_comp_fun = NULL;
			CUDA_CHECK_RETURN(
				cudaMemcpyFromSymbol(
					(void**) &my_comp_fun,
                    //TODO: Genericise this out
					(const void*) &HHSomaCompartment_simul_fxn_t,
					sizeof(void*),
					0,
					cudaMemcpyDeviceToHost
					)
				);
			copy_class._.MY_COMPARTMENT_SIMUL_CLASS_FXN = my_comp_fun;
		
			DEBUG_PRINTF("Copy Class comp fxn: %p\n", my_comp_fun);
		
			const struct MYRIADOBJECT_CLASS* super_class = (const struct MYRIADOBJECT_CLASS*) COMPARTMENT_CLASS;
			memcpy((void**) &copy_class_class->SUPERCLASS, &super_class->ONDEVICE_CLASS, sizeof(void*));
		}

		// This works because super methods rely on the given class'
		// semi-static superclass definition, not it's ->super attribute.
		// Note that we don't want to clobber, so we set it to 0.
		return SUPERCLASS_CUDAFY(COMPARTMENT_CLASS, (void*) &copy_class, 0);
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

const void* HHSOMACOMPARTMENT_OBJECT;
const void* HHSOMACOMPARTMENT_CLASS;

MYRIAD_FXN_METHOD_HEADER_GEN_NO_SUFFIX(DYNAMIC_INIT_FXN_RET, DYNAMIC_INIT_FXN_ARGS, HHSOMACOMPARTMENT_INIT_FXN_NAME)
//void initHHSomaCompartment(int init_cuda)
{
	// initCompartment(init_cuda);

	if (!HHSOMACOMPARTMENT_CLASS)
	{
		HHSOMACOMPARTMENT_CLASS =
			myriad_new(
				COMPARTMENT_CLASS,
				COMPARTMENT_CLASS,
				sizeof(struct HHSOMACOMPARTMENT_CLASS),
				myriad_cudafy, MYRIAD_CAT(HHSOMACOMPARTMENT_CLASS, MYRIAD_CAT(_, CUDAFY_FUN_NAME)),
				0
			);

		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_comp_c_t = myriad_cudafy((void*)HHSOMACOMPARTMENT_CLASS, 1);
			((struct MYRIADOBJECT_CLASS*) HHSOMACOMPARTMENT_CLASS)->ONDEVICE_CLASS = (struct MYRIADOBJECT_CLASS*) tmp_comp_c_t;
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(HHSOMACOMPARTMENT_CLASS, MYRIAD_CAT(_, DEV_T)),
					&tmp_comp_c_t,
					sizeof(struct HHSOMACOMPARTMENT_CLASS*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}

	if (!HHSOMACOMPARTMENT_OBJECT)
	{
		HHSOMACOMPARTMENT_OBJECT =
			myriad_new(
				HHSOMACOMPARTMENT_CLASS,
				COMPARTMENT_OBJECT,
				sizeof(struct HHSOMACOMPARTMENT_OBJECT),
				myriad_ctor, MYRIAD_CAT(HHSOMACOMPARTMENT_OBJECT, MYRIAD_CAT(_, CTOR_FUN_NAME)),
				myriad_dtor, MYRIAD_CAT(HHSOMACOMPARTMENT_OBJECT, MYRIAD_CAT(_, DTOR_FUN_NAME)),
				myriad_cudafy, MYRIAD_CAT(HHSOMACOMPARTMENT_OBJECT, MYRIAD_CAT(_, CUDAFY_FUN_NAME)),
				myriad_decudafy, MYRIAD_CAT(HHSOMACOMPARTMENT_OBJECT, MYRIAD_CAT(_, DECUDAFY_FUN_NAME)),
				SIMUL_FXN_NAME_D, MYRIAD_CAT(HHSOMACOMPARTMENT_OBJECT, MYRIAD_CAT(_, SIMUL_FXN_NAME_D)),
				0
			);

		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)HHSOMACOMPARTMENT_OBJECT, 1);
			((struct MYRIADOBJECT_CLASS*) HHSOMACOMPARTMENT_OBJECT)->ONDEVICE_CLASS = (struct MYRIADOBJECT_CLASS*) tmp_mech_t;
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(HHSOMACOMPARTMENT_OBJECT, MYRIAD_CAT(_, DEV_T)),
					&tmp_mech_t,
					sizeof(struct HHSOMACOMPARTMENT_OBJECT*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
}

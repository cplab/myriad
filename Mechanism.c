#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "myriad_debug.h"

#include "MyriadObject.h"
#include "Mechanism.h"
#include "Mechanism.cuh"

#include "Mechanism_meta.h"


///////////////////////////////
// Mechanism Super Overrides //
///////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, MECHANISM_OBJECT, CTOR_FUN_NAME)
{
	struct MECHANISM_OBJECT* _self = (struct MECHANISM_OBJECT*) SUPERCLASS_CTOR(MECHANISM_OBJECT, self, app);
	_self->COMPARTMENT_PREMECH_SOURCE_ID = va_arg(*app, unsigned int);

	return _self;
}

//////////////////////////
// Native Mechanism Fxn //
//////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(MECH_FXN_RET, MECH_FXN_ARGS, MECHANISM_OBJECT, MECH_FXN_NAME_C)
{
	const struct MECHANISM_OBJECT* self = (const struct MECHANISM_OBJECT*) _self;
	printf("My source id is %u\n", self->COMPARTMENT_PREMECH_SOURCE_ID);
	return 0.0;
}

MYRIAD_FXN_METHOD_HEADER_GEN_NO_SUFFIX(MECH_FXN_RET, MECH_FXN_ARGS, MECH_FXN_NAME_C)
{
	const struct MECHANISM_CLASS* OBJECTS_CLASS = (const struct MECHANISM_CLASS*) myriad_class_of(_self);
	assert(OBJECTS_CLASS->MY_MECHANISM_MECH_CLASS_FXN);
	return OBJECTS_CLASS->MY_MECHANISM_MECH_CLASS_FXN(_self, pre_comp, post_comp, dt, global_time, curr_step);
}

//TODO: GENERICISE THIS!
double super_mechanism_fxn(
	void* _class,
	void* _self,
    void* pre_comp,
    void* post_comp,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct MechanismClass* s_class=(const struct MechanismClass*) myriad_super(_class);
	assert(_self && s_class->m_mech_fxn);
	return s_class->m_mech_fxn(_self, pre_comp, post_comp, dt, global_time, curr_step);
}

////////////////////////////////////
// MechanismClass Super Overrides //
////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, MECHANISM_CLASS, CTOR_FUN_NAME)
{
	struct MECHANISM_CLASS* _self = (struct MECHANISM_CLASS*) SUPERCLASS_CTOR(MECHANISM_CLASS, self, app);

	voidf selector = NULL; selector = va_arg(*app, voidf);

	while (selector)
	{
		const voidf method = va_arg(*app, voidf);
		
		if (selector == (voidf) MECH_FXN_NAME_C)
		{
			*(voidf *) &_self->MY_MECHANISM_MECH_CLASS_FXN = method;
		}

		selector = va_arg(*app, voidf);
	}

	return _self;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, MECHANISM_CLASS, CUDAFY_FUN_NAME)
{
	#ifdef CUDA
	{
		// We know that we're actually a mechanism class
		struct MECHANISM_CLASS* my_class = (struct MECHANISM_CLASS*) _self;

		// Make a temporary copy-class because we need to change shit
		struct MECHANISM_CLASS copy_class = *my_class; // Assignment to stack avoids calloc/memcpy
		struct MYRIADOBJECT_CLASS* copy_class_class = (struct MYRIADOBJECT_CLASS*) &copy_class;

		// TODO: Find a better way to get function pointers for on-card functions
		MECH_FXN_NAME my_mech_fun = NULL;
		CUDA_CHECK_RETURN(
			cudaMemcpyFromSymbol(
				(void**) &my_mech_fun,
				(const void*) &MYRIAD_CAT(MECHANISM_OBJECT, _cuda_mechanism_fxn_t),
				sizeof(void*),
				0,
				cudaMemcpyDeviceToHost
				)
			);
		copy_class.MY_MECHANISM_MECH_CLASS_FXN = my_mech_fun;
		DEBUG_PRINTF("Copy Class mech fxn: %p\n", my_mech_fun);
	
		// !!!!!!!!! IMPORTANT !!!!!!!!!!!!!!
		// By default we clobber the copy_class_class' superclass with
		// the superclass' device_class' on-GPU address value. 
		// To avoid cloberring this value (e.g. if an underclass has already
		// clobbered it), the clobber flag should be 0.
		if (clobber)
		{
			const struct MECHANISM_CLASS* super_class = (const struct MECHANISM_CLASS*) MECHANISM_CLASS;
			memcpy((void**) &copy_class_class->SUPERCLASS, &super_class->ONDEVICE_CLASS, sizeof(void*));
		}

		// This works because super methods rely on the given class'
		// semi-static superclass definition, not it's ->super attribute.
		return super_cudafy(MECHANISM_CLASS, (void*) &copy_class, 0);
	}
	#else
	{
		return NULL;
	}
	#endif
}

/////////////////////////////////////
// Reference Object Initialization //
/////////////////////////////////////

const void *MECHANISM_CLASS, *MECHANISM_OBJECT;

void initMechanism(int init_cuda)
{
	if (!MECHANISM_CLASS)
	{
		MECHANISM_CLASS = 
			myriad_new(
				   MYRIADOBJECT_CLASS,
				   MYRIADOBJECT_CLASS,
				   sizeof(struct MECHANISM_CLASS),
				   myriad_ctor, MYRIAD_CAT(MECHANISM_CLASS, MYRIAD_CAT(_, CTOR_FUN_NAME)),
				   myriad_cudafy, MYRIAD_CAT(MECHANISM_CLASS, MYRIAD_CAT(_, CUDAFY_FUN_NAME)),
				   0
			);
		struct MYRIADOBJECT_OBJECT* mech_class_obj = NULL; mech_class_obj = (struct MYRIADOBJECT_OBJECT*) MECHANISM_CLASS;
		memcpy( (void**) &mech_class_obj->OBJECTS_CLASS, &MECHANISM_CLASS, sizeof(void*));

		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)MECHANISM_CLASS, 1);
			((struct MYRIADOBJECT_CLASS*) MECHANISM_CLASS)->ONDEVICE_CLASS = (struct MYRIADOBJECT_CLASS*) tmp_mech_c_t;
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(MECHANISM_CLASS, _dev_t),
					&tmp_mech_c_t,
					sizeof(struct MECHANISM_CLASS*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
	
	if (!MECHANISM_OBJECT)
	{
		MECHANISM_OBJECT = 
			myriad_new(
				   MECHANISM_CLASS,
				   MYRIADOBJECT_OBJECT,
				   sizeof(struct MECHANISM_OBJECT),
				   myriad_ctor, MYRIAD_CAT(MECHANISM_OBJECT, MYRIAD_CAT(_, CTOR_FUN_NAME)),
				   mechanism_fxn, MYRIAD_CAT(MECHANISM_OBJECT, _mechanism_fxn),
				   0
			);

		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)MECHANISM_OBJECT, 1);
			((struct MYRIADOBJECT_CLASS*) MECHANISM_OBJECT)->ONDEVICE_CLASS = (struct MYRIADOBJECT_CLASS*) tmp_mech_t;
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(MECHANISM_OBJECT, _dev_t),
					&tmp_mech_t,
					sizeof(struct MECHANISM_OBJECT*),
					0,
					cudaMemcpyHostToDevice
					)
				);

		}
		#endif
	}
	
}

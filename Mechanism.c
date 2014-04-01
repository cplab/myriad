#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "myriad_debug.h"

#include "MyriadObject.h"
#include "Mechanism.h"
#include "Mechanism.cuh"

///////////////////////////////
// Mechanism Super Overrides //
///////////////////////////////

static void* Mechanism_ctor(void* _self, va_list* app)
{
	struct Mechanism* self = (struct Mechanism*) super_ctor(Mechanism, _self, app);
	
	self->source_id = va_arg(*app, unsigned int);

	return self;
}

static int Mechanism_dtor(void* _self)
{
	// Don't need to free anything ourselves, just pass along to super
	return super_dtor(MyriadObject, _self);
}

static void* Mechanism_cudafy(void* _self, int clobber)
{
	//TODO: What value of clobber for non-class objects?
	return super_cudafy(Mechanism, _self, clobber); 
}

static void Mechanism_decudafy(void* _self, void* cuda_class)
{
	// Do nothing; assume source ID hasn't changed
	super_decudafy(MyriadObject, _self, cuda_class);
	return;
}

//////////////////////////
// Native Mechanism Fxn //
//////////////////////////

static double Mechanism_mechanism_fxn(
	void* _self,
    void* pre_comp,
    void* post_comp,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct Mechanism* self = (const struct Mechanism*) _self;
	printf("My source id is %u\n", self->source_id);
	return 0.0;
}

double mechanism_fxn(
	void* _self,
    void* pre_comp,
    void* post_comp,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct MechanismClass* m_class = (const struct MechanismClass*) myriad_class_of(_self);
	assert(m_class->m_mech_fxn);
	return m_class->m_mech_fxn(_self, pre_comp, post_comp, dt, global_time, curr_step);
}

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

static void* MechanismClass_ctor(void* _self, va_list* app)
{
	struct MechanismClass* self = (struct MechanismClass*) super_ctor(MechanismClass, _self, app);

	voidf selector = NULL; selector = va_arg(*app, voidf);

	while (selector)
	{
		const voidf method = va_arg(*app, voidf);
		
		if (selector == (voidf) mechanism_fxn)
		{
			*(voidf *) &self->m_mech_fxn = method;
		}

		selector = va_arg(*app, voidf);
	}

	return self;
}

static int MechanismClass_dtor(void* _self)
{
	// Undefined behavior, let superclass handle it
	return super_dtor(MyriadClass, _self);
}

static void* MechanismClass_cudafy(void* _self, int clobber)
{
	void* result = NULL;
	
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
	printf("Copy Class mech fxn: %p\n", my_mech_fun);
	
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
	result = super_cudafy(MechanismClass, (void*) &copy_class, 0);
	
	return result;
}

static void MechanismClass_decudafy(void* _self, void* cuda_class)
{
	// Undefined; let superclass worry about it
	super_decudafy(MyriadClass, _self, cuda_class);
	return;
}

/////////////////////////////////////
// Reference Object Initialization //
/////////////////////////////////////

const void *MechanismClass, *Mechanism;

void initMechanism(int init_cuda)
{
	if (!MechanismClass)
	{
		MechanismClass = 
			myriad_new(
				   MyriadClass,
				   MyriadClass,
				   sizeof(struct MechanismClass),
				   myriad_ctor, MechanismClass_ctor,
				   myriad_dtor, MechanismClass_dtor,
				   myriad_cudafy, MechanismClass_cudafy,
				   myriad_decudafy, MechanismClass_decudafy,
				   0
			);
		struct MyriadObject* mech_class_obj = NULL; mech_class_obj = (struct MyriadObject*) MechanismClass;
		memcpy( (void**) &mech_class_obj->m_class, &MechanismClass, sizeof(void*));

		// TODO: Additional checks for CUDA initialization
		if (init_cuda)
		{
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
		}
	}
	
	if (!Mechanism)
	{
		Mechanism = 
			myriad_new(
				   MechanismClass,
				   MyriadObject,
				   sizeof(struct Mechanism),
				   myriad_ctor, Mechanism_ctor,
				   myriad_dtor, Mechanism_dtor,
				   myriad_cudafy, Mechanism_cudafy,
				   mechanism_fxn, Mechanism_mechanism_fxn,
				   myriad_decudafy, Mechanism_decudafy,
				   0
			);

		if (init_cuda)
		{
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

		}
	}
	
}

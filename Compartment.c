#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "myriad_debug.h"

#include "MyriadObject.h"
#include "Compartment.h"
#include "Compartment.cuh"

/////////////////////////////////
// Compartment Super Overrides //
/////////////////////////////////

static void* Compartment_ctor(void* _self, va_list* app)
{
	struct Compartment* self = (struct Compartment*) super_ctor(Compartment, _self, app);
	
	self->id = va_arg(*app, unsigned int);

	return self;
}

//TODO: We might not actually need all these overrides if they're No-Ops; saves us a fxn call

static int Compartment_dtor(void* _self)
{
	// Passthrough to super since we didn't allocate anything ourselves
	return super_dtor(MyriadObject, _self);
}

static void* Compartment_cudafy(void* _self, int clobber)
{
	//TODO: What value of clobber for non-class objects?
	return super_cudafy(Compartment, _self, clobber); 
}

static void Compartment_decudafy(void* _self, void* cuda_self)
{
	// Passthrough to super; we don't need to do anything
	super_decudafy(MyriadObject, _self, cuda_self);
	return;
}

//////////////////////////////////////
// Native Functions Implementations //
//////////////////////////////////////

static void Compartment_simul_fxn(
	void* _self,
	void** network,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct Compartment* self = (const struct Compartment*) _self;
	printf("My id is %u\n", self->id);
	return;
}

void simul_fxn(
	void* _self,
	void** network,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct CompartmentClass* m_class = 
		(const struct CompartmentClass*) myriad_class_of((void*) _self);
	assert(m_class->m_comp_fxn);
	return m_class->m_comp_fxn(_self, network, dt, global_time, curr_step);
}

void super_simul_fxn(
	void* _class,
	void* _self,
	void** network,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct CompartmentClass* s_class=(const struct CompartmentClass*) myriad_super(_class);
	assert(_self && s_class->m_comp_fxn);
	return s_class->m_comp_fxn(_self, network, dt, global_time, curr_step);
}

//////////////////////////////////////
// CompartmentClass Super Overrides //
//////////////////////////////////////

static void* CompartmentClass_ctor(void* _self, va_list* app)
{
	struct CompartmentClass* self = (struct CompartmentClass*) super_ctor(CompartmentClass, _self, app);

	voidf selector = NULL; selector = va_arg(*app, voidf);

	while (selector)
	{
		const voidf method = va_arg(*app, voidf);
		
		if (selector == (voidf) simul_fxn)
		{
			*(voidf *) &self->m_comp_fxn = method;
		}

		selector = va_arg(*app, voidf);
	}

	return self;
}

static int CompartmentClass_dtor(void* _self)
{
	// Technically this is undefined behavior but the superclass can handle that
	return super_dtor(MyriadClass, _self);
}

static void* CompartmentClass_cudafy(void* _self, int clobber)
{
	void* result = NULL;
	
    // We know what class we are
	struct CompartmentClass* my_class = (struct CompartmentClass*) _self;

	// Make a temporary copy-class because we need to change shit
	struct CompartmentClass copy_class = *my_class;
	struct MyriadClass* copy_class_class = (struct MyriadClass*) &copy_class;

	// TODO: Find a better way to get function pointers for on-card functions
	compartment_simul_fxn_t my_comp_fun = NULL;
	CUDA_CHECK_RETURN(
		cudaMemcpyFromSymbol(
			(void**) &my_comp_fun,
			(const void*) &Compartment_cuda_compartment_fxn_t,
			sizeof(void*),
			0,
			cudaMemcpyDeviceToHost
			)
		);
	copy_class.m_comp_fxn = my_comp_fun;
	
	DEBUG_PRINTF("Copy Class comp fxn: %p\n", my_comp_fun);
	
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
	result = super_cudafy(CompartmentClass, (void*) &copy_class, 0);
	
	return result;
}

static void CompartmentClass_decudafy(void* _self, void* cuda_self)
{
	// Undefined; let the superclass yell at them
	super_decudafy(MyriadClass, _self, cuda_self);
	return;
}

///////////////////////////
// Object Initialization //
///////////////////////////

const void *CompartmentClass, *Compartment;

void initCompartment(int init_cuda)
{
	if (!CompartmentClass)
	{
		CompartmentClass = 
			myriad_new(
				   MyriadClass,
				   MyriadClass,
				   sizeof(struct CompartmentClass),
				   myriad_ctor, CompartmentClass_ctor,
				   myriad_dtor, CompartmentClass_dtor,
				   myriad_cudafy, CompartmentClass_cudafy,
				   myriad_decudafy, CompartmentClass_decudafy,
				   0
			);
		struct MyriadObject* mech_class_obj = (struct MyriadObject*) CompartmentClass;
		memcpy( (void**) &mech_class_obj->m_class, &CompartmentClass, sizeof(void*));

		// TODO: Additional checks for CUDA initialization
		if (init_cuda)
		{
			void* tmp_comp_c_t = myriad_cudafy((void*)CompartmentClass, 1);
			((struct MyriadClass*) CompartmentClass)->device_class = (struct MyriadClass*) tmp_comp_c_t;
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &CompartmentClass_dev_t,
					&tmp_comp_c_t,
					sizeof(struct CompartmentClass*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
	}
	
	if (!Compartment)
	{
		Compartment = 
			myriad_new(
				   CompartmentClass,
				   MyriadObject,
				   sizeof(struct Compartment),
				   myriad_ctor, Compartment_ctor,
				   myriad_dtor, Compartment_dtor,
				   myriad_cudafy, Compartment_cudafy,
				   simul_fxn, Compartment_simul_fxn,
   				   myriad_decudafy, Compartment_decudafy,
				   0
			);

		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)Compartment, 1);
			((struct MyriadClass*) Compartment)->device_class = (struct MyriadClass*) tmp_mech_t;
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &Compartment_dev_t,
					&tmp_mech_t,
					sizeof(struct Compartment*),
					0,
					cudaMemcpyHostToDevice
					)
				);

		}
	}
	
}


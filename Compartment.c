#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>


#include "myriad_debug.h"

#include "MyriadObject.h"
#include "Compartment.h"
#include "Compartment.cuh"
#include "Mechanism.h"

/////////////////////////////////
// Compartment Super Overrides //
/////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, COMPARTMENT_OBJECT, CTOR_FUN_NAME)
{
	struct COMPARTMENT_OBJECT* _self = (struct COMPARTMENT_OBJECT*) SUPERCLASS_CTOR(COMPARTMENT_OBJECT, self, app);
	
	_self->ID = va_arg(*app, unsigned int);
	_self->NUM_MECHS = va_arg(*app, unsigned int);
	_self->MY_MECHS = va_arg(*app, struct MECHANISM_OBJECT**);

	return _self;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, COMPARTMENT_OBJECT, CUDAFY_FUN_NAME)
{
	#ifdef CUDA
	{
		struct COMPARTMENT_OBJECT* self = (struct COMPARTMENT_OBJECT*) _self;

		// Copies entire struct onto independent, static, on-stack copy
		const size_t my_size = myriad_size_of(_self);
		struct COMPARTMENT_OBJECT* copy_comp = (struct COMPARTMENT_OBJECT*) calloc(1, my_size);
		memcpy(copy_comp, self, my_size);

		if (copy_comp->MY_MECHS != NULL)
		{
			// We'll assume that mechanism pointers already point to stuff on GPU,
			// we're just copying the values over (i.e. "shallow copy")

			CUDA_CHECK_RETURN(
				cudaMalloc( 
					(void**) &copy_comp->MY_MECHS, 
					copy_comp->NUM_MECHS * sizeof(struct COMPARTMENT_OBJECT*)
					)
				);

			CUDA_CHECK_RETURN(
				cudaMemcpy(
					copy_comp->MY_MECHS,
					self->MY_MECHS,
					copy_comp->NUM_MECHS * sizeof(struct COMPARTMENT_OBJECT*),
					cudaMemcpyHostToDevice
					)
				);
		}
		// @TODO: Should we really be passing a pointer to something on our stack?
		return SUPERCLASS_CUDAFY(MYRIADOBJECT_OBJECT, copy_comp, clobber);
	}
	#else
	{
		return NULL;
	}
	#endif
}

//////////////////////////////////////
// Native Functions Implementations //
//////////////////////////////////////

// Simulate function
static MYRIAD_FXN_METHOD_HEADER_GEN(SIMUL_FXN_RET, SIMUL_FXN_ARGS, COMPARTMENT_OBJECT, INDIVIDUAL_SIMUL_FXN_NAME)
/*static void Compartment_simul_fxn(
	void* _self,
	void** network,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	)*/
{
	const struct COMPARTMENT_OBJECT* _self = (const struct COMPARTMENT_OBJECT*) self;
	printf("My id is %u\n", _self->ID);
	printf("My num_mechs is %u\n", _self->NUM_MECHS);
	return;
}

MYRIAD_FXN_METHOD_HEADER_GEN_NO_SUFFIX(SIMUL_FXN_RET, SIMUL_FXN_ARGS, INDIVIDUAL_SIMUL_FXN_NAME)
/*void simul_fxn(
	void* _self,
	void** network,
	const double dt,
	const double global_time,
	const unsigned int curr_step
	)*/
{
	const struct COMPARTMENT_CLASS* OBJECTS_CLASS = 
		(const struct COMPARTMENT_CLASS*) myriad_class_of((void*) self);
	assert(OBJECTS_CLASS->MY_COMPARTMENT_SIMUL_CLASS_FXN);
	OBJECTS_CLASS->MY_COMPARTMENT_SIMUL_CLASS_FXN(self, network, dt, global_time, curr_step);
}

MYRIAD_FXN_METHOD_HEADER_GEN(SIMUL_FXN_RET, SUPER_SIMUL_FXN_ARGS, SUPERCLASS, SIMUL_FXN_NAME_D)
/*void super_simul_fxn(
	void* _class,
	void* _self,
	void** network,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)*/
{
	const struct COMPARTMENT_CLASS* s_class=(const struct COMPARTMENT_CLASS*) myriad_super(_class);
	assert(_self && s_class->MY_COMPARTMENT_SIMUL_CLASS_FXN);
	s_class->MY_COMPARTMENT_SIMUL_CLASS_FXN(_self, network, dt, global_time, curr_step);
}

// Add mechanism function

static MYRIAD_FXN_METHOD_HEADER_GEN(ADDMECH_FXN_RET, ADDMECH_FXN_ARGS, COMPARTMENT_OBJECT, INDIVIDUAL_ADDMECH_FXN_NAME)
{
	if (self == NULL || mechanism == NULL)
	{
		DEBUG_PRINTF("Cannot add NULL mechanism/add to NULL compartment.\n");
		return EXIT_FAILURE;
	}

	struct COMPARTMENT_OBJECT* _self = (struct COMPARTMENT_OBJECT*) self;
	struct MECHANISM_OBJECT* mech = (struct MECHANISM_OBJECT*) mechanism;
	
	_self->NUM_MECHS++;
	_self->MY_MECHS = (struct MECHANISM_OBJECT**) realloc(_self->MY_MECHS, sizeof(struct MECHANISM_OBJECT*) * _self->NUM_MECHS);

	if (_self->MY_MECHS == NULL)
	{
		DEBUG_PRINTF("Could not reallocate mechanisms array.\n");
		return EXIT_FAILURE;
	}

	_self->MY_MECHS[_self->NUM_MECHS-1] = mech;

	return EXIT_SUCCESS;
}

MYRIAD_FXN_METHOD_HEADER_GEN_NO_SUFFIX(ADDMECH_FXN_RET, ADDMECH_FXN_ARGS, ADDMECH_FXN_NAME_D)
{
	const struct COMPARTMENT_CLASS* OBJECTS_CLASS = 
		(const struct COMPARTMENT_CLASS*) myriad_class_of((void*) self);
	assert(OBJECTS_CLASS->MY_COMPARTMENT_ADDMECH_CLASS_FXN);
	return OBJECTS_CLASS->MY_COMPARTMENT_ADDMECH_CLASS_FXN(self, mechanism);
}

MYRIAD_FXN_METHOD_HEADER_GEN(ADDMECH_FXN_RET, SUPER_ADDMECH_FXN_ARGS, SUPERCLASS, ADDMECH_FXN_NAME_D)
{
	const struct COMPARTMENT_CLASS* s_class=(const struct COMPARTMENT_CLASS*) myriad_super(_class);
	assert(self && s_class->MY_COMPARTMENT_ADDMECH_CLASS_FXN);
	return s_class->MY_COMPARTMENT_ADDMECH_CLASS_FXN(self, mechanism);
}

//////////////////////////////////////
// CompartmentClass Super Overrides //
//////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, COMPARTMENT_CLASS, CTOR_FUN_NAME)
{
	struct COMPARTMENT_CLASS* _self = (struct COMPARTMENT_CLASS*) SUPERCLASS_CTOR(COMPARTMENT_CLASS, self, app);

	voidf selector = NULL; selector = va_arg(*app, voidf);

	while (selector)
	{
		const voidf method = va_arg(*app, voidf);
		
		if (selector == (voidf) SIMUL_FXN_NAME_D)
		{
			*(voidf *) &_self->MY_COMPARTMENT_SIMUL_CLASS_FXN = method;
		} else if (selector == (voidf) add_mechanism) {
			*(voidf *) &_self->MY_COMPARTMENT_ADDMECH_CLASS_FXN = method;
		}

		selector = va_arg(*app, voidf);
	}

	return _self;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, COMPARTMENT_CLASS, CUDAFY_FUN_NAME)
{
    #ifdef CUDA
    {
        // We know what class we are
        struct COMPARTMENT_CLASS* my_class = (struct COMPARTMENT_CLASS*) _self;

        // Make a temporary copy-class because we need to change shit
        struct COMPARTMENT_CLASS copy_class = *my_class;
        struct MYRIADOBJECT_CLASS* copy_class_class = (struct MYRIADOBJECT_CLASS*) &copy_class;

        // !!!!!!!!! IMPORTANT !!!!!!!!!!!!!!
        // By default we clobber the copy_class_class' superclass with
        // the superclass' device_class' on-GPU address value. 
        // To avoid cloberring this value (e.g. if an underclass has already
        // clobbered it), the clobber flag should be 0.
        if (clobber)
        {
            // TODO: Find a better way to get function pointers for on-card functions
            SIMUL_FXN_NAME_T my_comp_fun = NULL;
            CUDA_CHECK_RETURN(
                cudaMemcpyFromSymbol(
                    (void**) &my_comp_fun,
                    //TOOD: Generecise this out
                    (const void*) &Compartment_cuda_compartment_fxn_t,
                    sizeof(void*),
                    0,
                    cudaMemcpyDeviceToHost
                    )
                );
            copy_class.MY_COMPARTMENT_SIMUL_CLASS_FXN = my_comp_fun;
		
            DEBUG_PRINTF("Copy Class comp fxn: %p\n", my_comp_fun);
		
            const struct MYRIADOBJECT_CLASS* super_class = (const struct MYRIADOBJECT_CLASS*) MYRIADOBJECT_CLASS;
            memcpy((void**) &copy_class_class->SUPERCLASS, &super_class->ONDEVICE_CLASS, sizeof(void*));
        }

        // This works because super methods rely on the given class'
        // semi-static superclass definition, not it's ->super attribute.
        return SUPERCLASS_CUDAFY(COMPARTMENT_CLASS, (void*) &copy_class, 0);
    }
    #else
    {
        return NULL;
    }
    #endif
}

///////////////////////////
// Object Initialization //
///////////////////////////

const void * COMPARTMENT_CLASS, * COMPARTMENT_OBJECT;

MYRIAD_FXN_METHOD_HEADER_GEN_NO_SUFFIX(DYNAMIC_INIT_FXN_RET, DYNAMIC_INIT_FXN_ARGS, COMPARTMENT_INIT_FXN_NAME)
{
	if (!COMPARTMENT_CLASS)
	{
		COMPARTMENT_CLASS = 
			myriad_new(
				   MYRIADOBJECT_CLASS,
				   MYRIADOBJECT_CLASS,
				   sizeof(struct COMPARTMENT_CLASS),
				   myriad_ctor, MYRIAD_CAT(COMPARTMENT_CLASS, MYRIAD_CAT(_, CTOR_FUN_NAME)),
				   myriad_cudafy, MYRIAD_CAT(COMPARTMENT_CLASS, MYRIAD_CAT(_, CUDAFY_FUN_NAME)),
				   0
			);
		struct MYRIADOBJECT_OBJECT* mech_class_obj = (struct MYRIADOBJECT_OBJECT*) COMPARTMENT_CLASS;
		memcpy( (void**) &mech_class_obj->OBJECTS_CLASS, &COMPARTMENT_CLASS, sizeof(void*));

		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_comp_c_t = myriad_cudafy((void*)COMPARTMENT_CLASS, 1);
			((struct MYRIADOBJECT_CLASS*) COMPARTMENT_CLASS)->ONDEVICE_CLASS = (struct MYRIADOBJECT_CLASS*) tmp_comp_c_t;
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(COMPARTMENT_CLASS, MYRIAD_CAT(_, DEV_T)),
					&tmp_comp_c_t,
					sizeof(struct COMPARTMENT_CLASS*),
					0,
					cudaMemcpyHostToDevice
					)
				);
		}
		#endif
	}
	
	if (!COMPARTMENT_OBJECT)
	{
		COMPARTMENT_OBJECT = 
			myriad_new(
				   COMPARTMENT_CLASS,
				   MYRIADOBJECT_OBJECT,
				   sizeof(struct COMPARTMENT_OBJECT),
				   myriad_ctor, MYRIAD_CAT(COMPARTMENT_OBJECT, MYRIAD_CAT(_, CTOR_FUN_NAME)),
				   myriad_cudafy, MYRIAD_CAT(COMPARTMENT_OBJECT, MYRIAD_CAT(_, CUDAFY_FUN_NAME)),
				   SIMUL_FXN_NAME_D, MYRIAD_CAT(COMPARTMENT_OBJECT, MYRIAD_CAT(_, INDIVIDUAL_SIMUL_FXN_NAME)), 
				   ADDMECH_FXN_NAME_D, MYRIAD_CAT(COMPARTMENT_OBJECT, MYRIAD_CAT(_, INDIVIDUAL_ADDMECH_FXN_NAME)),
				   0
			);

		#ifdef CUDA
		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)COMPARTMENT_OBJECT, 1);
			((struct MYRIADOBJECT_CLASS*) COMPARTMENT_OBJECT)->ONDEVICE_CLASS = (struct MYRIADOBJECT_CLASS*) tmp_mech_t;
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					(const void*) &MYRIAD_CAT(COMPARTMENT_OBJECT, MYRIAD_CAT(_, DEV_T)),
					&tmp_mech_t,
					sizeof(struct COMPARTMENT_OBJECT*),
					0,
					cudaMemcpyHostToDevice
					)
				);

		}
		#endif
	}
	
}


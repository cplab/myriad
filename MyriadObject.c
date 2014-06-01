#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>
#include <string.h>

#include "myriad_debug.h"

#include "MyriadObject.h"
#include "MyriadObject.cuh"

////////////////////////////////////////////
// Forward declaration for static methods //
////////////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, MYRIADOBJECT_OBJECT, CTOR_FUN_NAME);
static MYRIAD_FXN_METHOD_HEADER_GEN(DTOR_FUN_RET, DTOR_FUN_ARGS, MYRIADOBJECT_OBJECT, DTOR_FUN_NAME);
static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, MYRIADOBJECT_OBJECT, CUDAFY_FUN_NAME);
static MYRIAD_FXN_METHOD_HEADER_GEN(DECUDAFY_FUN_RET, DECUDAFY_FUN_ARGS, MYRIADOBJECT_OBJECT, DECUDAFY_FUN_NAME);

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, MYRIADOBJECT_CLASS, CTOR_FUN_NAME);
static MYRIAD_FXN_METHOD_HEADER_GEN(DTOR_FUN_RET, DTOR_FUN_ARGS, MYRIADOBJECT_CLASS, DTOR_FUN_NAME);
static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, MYRIADOBJECT_CLASS, CUDAFY_FUN_NAME);
static MYRIAD_FXN_METHOD_HEADER_GEN(DECUDAFY_FUN_RET, DECUDAFY_FUN_ARGS, MYRIADOBJECT_CLASS, DECUDAFY_FUN_NAME);

///////////////////////////////////////////////////////
// Static initalization for new()/classof() purposes //
///////////////////////////////////////////////////////

// Static, on-stack initialization of MyriadObject and MyriadClass classes
// Necessary because of circular dependencies (see comments below)
static struct MYRIADOBJECT_CLASS object[] =
{
	// MyriadObject "anonymous" class
    {
        { object + 1 },              // MyriadClass is it's class
        object,                      // Superclass is itself (MyriadObject)
        NULL,                        // No device class by default
        sizeof(struct MYRIADOBJECT_OBJECT), // Size is effectively of pointer
		MYRIAD_CAT(MYRIADOBJECT_OBJECT, MYRIAD_CAT(_, CTOR_FUN_NAME)),           // Non-class constructor
		MYRIAD_CAT(MYRIADOBJECT_OBJECT, MYRIAD_CAT(_, DTOR_FUN_NAME)),           // Object destructor
		MYRIAD_CAT(MYRIADOBJECT_OBJECT, MYRIAD_CAT(_, CUDAFY_FUN_NAME)),         // Gets on device as an object
		MYRIAD_CAT(MYRIADOBJECT_OBJECT, MYRIAD_CAT(_, DECUDAFY_FUN_NAME)),       // In-place update of CPU object using GPU object
    },
	// MyriadClass class
    {
        { object + 1 },             // MyriadClass is it's class
        object,                     // Superclass is MyriadObject (a Class is an Object)
        NULL,                       // No device class by default
        sizeof(struct MYRIADOBJECT_CLASS), // Size includes methods, embedded MyriadObject
		MYRIAD_CAT(MYRIADOBJECT_CLASS, MYRIAD_CAT(_, CTOR_FUN_NAME)),           // Constructor allows for prototype classes
		MYRIAD_CAT(MYRIADOBJECT_CLASS, MYRIAD_CAT(_, DTOR_FUN_NAME)),           // Class destructor (No-Op, undefined behavior)
		MYRIAD_CAT(MYRIADOBJECT_CLASS, MYRIAD_CAT(_, CUDAFY_FUN_NAME)),         // Cudafication to avoid static init for extensions
		MYRIAD_CAT(MYRIADOBJECT_CLASS, MYRIAD_CAT(_, DECUDAFY_FUN_NAME)),       // No-Op; DeCUDAfying a class is undefined behavior
    }
};

// Pointers to static class definition for new()/super()/classof() purposes
const void* MYRIADOBJECT_OBJECT = object;
const void* MYRIADOBJECT_CLASS = object + 1;

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, MYRIADOBJECT_OBJECT, CTOR_FUN_NAME)
{
    return self;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(DTOR_FUN_RET, DTOR_FUN_ARGS, MYRIADOBJECT_OBJECT, DTOR_FUN_NAME)
{
	free(self);
	return EXIT_SUCCESS;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, MYRIADOBJECT_OBJECT, CUDAFY_FUN_NAME)
{
	#ifdef CUDA
	{
		struct MYRIADOBJECT_OBJECT* self = (struct MYRIADOBJECT_OBJECT*) self_obj;
		void* n_dev_obj = NULL;
		size_t my_size = myriad_size_of(self);

		const struct MYRIADOBJECT_CLASS* tmp = self->OBJECTS_CLASS;
		self->OBJECTS_CLASS = self->OBJECTS_CLASS->ONDEVICE_CLASS;

		CUDA_CHECK_RETURN(cudaMalloc(&n_dev_obj, my_size));

		CUDA_CHECK_RETURN(
			cudaMemcpy(
				n_dev_obj,
				self,
				my_size,
				cudaMemcpyHostToDevice
				)
			);

		self->OBJECTS_CLASS = tmp;

		return n_dev_obj;
	}
	#else
	{
		return NULL;
	}
	#endif
}

static MYRIAD_FXN_METHOD_HEADER_GEN(DECUDAFY_FUN_RET, DECUDAFY_FUN_ARGS, MYRIADOBJECT_OBJECT, DECUDAFY_FUN_NAME)
{
	// We assume (for now) that the class hasn't changed on the GPU.
	// This makes this effectively a no-op since nothing gets copied back
	return;
}

//////////////////////////////////////////////
// MyriadClass-specific static methods //
//////////////////////////////////////////////

static MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, MYRIADOBJECT_CLASS, CTOR_FUN_NAME)
{
    struct MYRIADOBJECT_CLASS* _self = (struct MYRIADOBJECT_CLASS*) self;
    const size_t offset = offsetof(struct MYRIADOBJECT_CLASS, my_ctor);

    _self->SUPERCLASS = va_arg(*app, struct MYRIADOBJECT_CLASS*);
    _self->OBJECTS_SIZE = va_arg(*app, size_t);

    assert(_self->SUPERCLASS);
	
	/*
	 * MASSIVE TODO:
	 * 
	 * Since this is generics-based we want to be able to have default behavior for classes
	 * that don't want to specify their own overrides; we probably then need to change this
	 * memcpy to account for ALL the methods, not just the ones we like.
	 * 
	 * Solution: Make it absolutely sure if we're memcpying ALL the methods.
	 */
	// Memcopies MyriadObject cudafy methods onto self (in case defaults aren't set)
    memcpy((char*) _self + offset,
		   (char*) _self->SUPERCLASS + offset,
		   myriad_size_of(_self->SUPERCLASS) - offset);

    va_list ap;
    va_copy(ap, *app);

    voidf selector = NULL; selector = va_arg(ap, voidf);

    while (selector)
    {
        const voidf curr_method = va_arg(ap, voidf);
    
        if (selector == (voidf) myriad_ctor)
        {
            *(voidf *) &_self->CONSTRUCTOR = curr_method;
        } else if (selector == (voidf) myriad_cudafy) {
			*(voidf *) &_self->CUDAFIER = curr_method;
		} else if (selector == (voidf) myriad_dtor) {
			*(voidf *) &_self->DESTRUCTOR = curr_method;
		} else if (selector == (voidf) myriad_decudafy) {
			*(voidf *) &_self->DECUDAFIER = curr_method;
		}
		
		selector = va_arg(ap, voidf);
    }

    return _self;
}

static MYRIAD_FXN_METHOD_HEADER_GEN(DTOR_FUN_RET, DTOR_FUN_ARGS, MYRIADOBJECT_CLASS, DTOR_FUN_NAME)
{
	fprintf(stderr, "Destroying a Class is undefined behavior.\n");
	return EXIT_FAILURE;
}

// IMPORTANT: This is, ironically, for external classes' use only, since our 
// own initialization for MyriadClass is static and handled by initCUDAObjects
static MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, MYRIADOBJECT_CLASS, CUDAFY_FUN_NAME)
{
	/*
	 * Invariants/Expectations: 
	 *
	 * A) The class we're given (_self) is fully initialized on the CPU
	 * B) _self->device_class == NULL, will receive this fxn's result
	 * C) _self->super has been set with (void*) SuperClass->device_class
	 *
	 * The problem here is that we're currently ignoring anything the 
	 * extended class passes up at us through super_, and so we're only
	 * copying the c_class struct, not the rest of the class. To solve this,
	 * what we need to do is to:
	 *
	 * 1) Memcopy the ENTIRETY of the old class onto a new heap pointer
	 *     - This works because the extended class has already made any 
	 *       and all of their pointers/functions CUDA-compatible.
	 * 2) Alter the "top-part" of the copied-class to go to CUDA
	 *     - cudaMalloc the future location of the class on the device
	 *     - Set our internal object's class pointer to that location
	 * 3) Copy our copied-class to the device
	 * 3a) Free our copied-class
	 * 4) Return the device pointer to whoever called us
	 *
	 * Q: How do we keep track of on-device super class?
	 * A: We take it on good faith that the under class has set their super class
	 *    to be the visible SuperClass->device_class.
	 */
	#ifdef CUDA
	{
		struct MYRIADOBJECT_CLASS* self = (struct MYRIADOBJECT_CLASS*) _self;

		const struct MYRIADOBJECT_CLASS* dev_class = NULL;

		const size_t class_size = myriad_size_of(self); // DO NOT USE sizeof(struct MyriadClass)!

		// Allocate space for new class on the card
		CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_class, class_size));
	
		// Memcpy the entirety of the old class onto a new CPU heap pointer
		const struct MYRIADOBJECT_CLASS* class_cpy = (const struct MYRIADOBJECT_CLASS*) calloc(1, class_size);
		memcpy((void*)class_cpy, _self, class_size);

		// Embedded object's class set to our GPU class; this is unaffected by $clobber
		memcpy((void*)&class_cpy->_.OBJECTS_CLASS, &dev_class, sizeof(void*)); 

		CUDA_CHECK_RETURN(
			cudaMemcpy(
				(void*)dev_class,
				class_cpy,
				class_size,
				cudaMemcpyHostToDevice
				)
			);

		free((void*)class_cpy); // Can safely free since underclasses get nothing
		
		return (void*) dev_class;
	}
	#else
	{
		return NULL;
	}
	#endif
}

static MYRIAD_FXN_METHOD_HEADER_GEN(DECUDAFY_FUN_RET, DECUDAFY_FUN_ARGS, MYRIADOBJECT_CLASS, DECUDAFY_FUN_NAME)
{
	fprintf(stderr, "De-CUDAfying a class is undefined behavior. Aborted.\n");
	return;
}

/////////////////////////////////////
// Object management and Selectors //
/////////////////////////////////////

//----------------------------
//            New
//----------------------------

// Much of the following functions have been assumed to not need full
// genericisation as they shouldn't be touched by anyone.
void* myriad_new(const void* _class, ...)
{
    const struct MYRIADOBJECT_CLASS* prototype_class = (const struct MYRIADOBJECT_CLASS*) _class;
    struct MYRIADOBJECT_OBJECT* curr_obj;
    va_list ap;

    assert(prototype_class && prototype_class->OBJECTS_SIZE);
    
    curr_obj = (struct MYRIADOBJECT_OBJECT*) calloc(1, prototype_class->OBJECTS_SIZE);
    assert(curr_obj);

    curr_obj->OBJECTS_CLASS = prototype_class;

    va_start(ap, _class);
    curr_obj = (struct MYRIADOBJECT_OBJECT*) myriad_ctor(curr_obj, &ap);
    va_end(ap);
	
    return curr_obj;
}

//----------------------------
//         Class Of
//----------------------------

const void* myriad_class_of(const void* _self)
{
    const struct MYRIADOBJECT_OBJECT* self = (const struct MYRIADOBJECT_OBJECT*) _self;
    return self->OBJECTS_CLASS;
}

//----------------------------
//         Size Of
//----------------------------

size_t myriad_size_of(const void* _self)
{
    const struct MYRIADOBJECT_CLASS* OBJECTS_CLASS = (const struct MYRIADOBJECT_CLASS*) myriad_class_of(_self);

    return OBJECTS_CLASS->OBJECTS_SIZE;
}

//----------------------------
//         Is A
//----------------------------

int myriad_is_a(const void* _self, const struct MYRIADOBJECT_CLASS* OBJECTS_CLASS)
{
    return _self && myriad_class_of(_self) == OBJECTS_CLASS;
}

//----------------------------
//          Is Of
//----------------------------

int myriad_is_of(const void* _self, const struct MYRIADOBJECT_CLASS* OBJECTS_CLASS)
{
    if (_self)
    {   
        const struct MYRIADOBJECT_CLASS * myClass = (const struct MYRIADOBJECT_CLASS*) myriad_class_of(_self);

        if (OBJECTS_CLASS != MYRIADOBJECT_OBJECT)
        {
            while (myClass != OBJECTS_CLASS)
            {
                if (myClass != MYRIADOBJECT_OBJECT)
                {
                    myClass = (const struct MYRIADOBJECT_CLASS*) myriad_super(myClass);
                } else {
                    return 0;
                }
            }
        }

        return 1;
    }

    return 0;
}

//------------------------------
//   Object Built-in Generics
//------------------------------

MYRIAD_FXN_METHOD_HEADER_GEN(CTOR_FUN_RET, CTOR_FUN_ARGS, myriad, CTOR_FUN_NAME)
{
    const struct MYRIADOBJECT_CLASS* OBJECTS_CLASS = (const struct MYRIADOBJECT_CLASS*) myriad_class_of(self);

    assert(OBJECTS_CLASS->CONSTRUCTOR);
    return OBJECTS_CLASS->CONSTRUCTOR(self, app);
}

MYRIAD_FXN_METHOD_HEADER_GEN(DTOR_FUN_RET, DTOR_FUN_ARGS, myriad, DTOR_FUN_NAME)
{
    const struct MYRIADOBJECT_CLASS* OBJECTS_CLASS = (const struct MYRIADOBJECT_CLASS*) myriad_class_of(self);

    assert(OBJECTS_CLASS->DESTRUCTOR);
    return OBJECTS_CLASS->DESTRUCTOR(self);
}

MYRIAD_FXN_METHOD_HEADER_GEN(CUDAFY_FUN_RET, CUDAFY_FUN_ARGS, myriad, CUDAFY_FUN_NAME)
{
    const struct MYRIADOBJECT_CLASS* OBJECTS_CLASS = (const struct MYRIADOBJECT_CLASS*) myriad_class_of(_self);

	assert(OBJECTS_CLASS->CUDAFIER);
	return OBJECTS_CLASS->CUDAFIER(_self, clobber);
}

MYRIAD_FXN_METHOD_HEADER_GEN(DECUDAFY_FUN_RET, DECUDAFY_FUN_ARGS, myriad, DECUDAFY_FUN_NAME)
{
	const struct MYRIADOBJECT_CLASS* OBJECTS_CLASS = (const struct MYRIADOBJECT_CLASS*) myriad_class_of(_self);
	
	assert(OBJECTS_CLASS->DECUDAFIER);
	OBJECTS_CLASS->DECUDAFIER(_self, cuda_self);
	return;
}

///////////////////////////////
// Super and related methods //
///////////////////////////////

extern MYRIAD_FXN_METHOD_HEADER_GEN(SUPER_FUN_RET, SUPER_FUN_ARGS, myriad, SUPER_FUN_NAME)
{
    const struct MYRIADOBJECT_CLASS* self = (const struct MYRIADOBJECT_CLASS*) _self;

    assert(self && self->SUPERCLASS);
    return self->SUPERCLASS;
}

extern MYRIAD_FXN_METHOD_HEADER_GEN(SUPERCTOR_FUN_RET, SUPERCTOR_FUN_ARGS, SUPER_FUN_NAME, SUPERCTOR_FUN_NAME)
{
    const struct MYRIADOBJECT_CLASS* superclass = (const struct MYRIADOBJECT_CLASS*) myriad_super(_class);

    assert(_self && superclass->CONSTRUCTOR);
    return superclass->CONSTRUCTOR(_self, app);
}

MYRIAD_FXN_METHOD_HEADER_GEN(SUPERDTOR_FUN_RET, SUPERDTOR_FUN_ARGS, SUPER_FUN_NAME, SUPERDTOR_FUN_NAME)
{
	const struct MYRIADOBJECT_CLASS* superclass = (const struct MYRIADOBJECT_CLASS*) myriad_super(_class);

	assert(_self && superclass->DESTRUCTOR);
	return superclass->DESTRUCTOR(_self);
}

MYRIAD_FXN_METHOD_HEADER_GEN(SUPERCUDAFY_FUN_RET, SUPERCUDAFY_FUN_ARGS, SUPER_FUN_NAME, SUPERCUDAFY_FUN_NAME)
{
	const struct MYRIADOBJECT_CLASS* superclass = (const struct MYRIADOBJECT_CLASS*) myriad_super(_class);
	assert(_self && superclass->CUDAFIER);
	return superclass->CUDAFIER(_self, clobber);
}

MYRIAD_FXN_METHOD_HEADER_GEN(SUPERDECUDAFY_FUN_RET, SUPERDECUDAFY_FUN_ARGS, SUPER_FUN_NAME, SUPERDECUDAFY_FUN_NAME)
{
	const struct MYRIADOBJECT_CLASS* superclass = (const struct MYRIADOBJECT_CLASS*) myriad_super(_class);
	assert(_self && superclass->DECUDAFIER);
	superclass->DECUDAFIER(_self, cuda_self);
	return;
}

///////////////////////////////////
//   CUDA Object Initialization  //
///////////////////////////////////

int initCUDAObjects()
{
	// Can't initialize if there be no CUDA
	#ifdef CUDA
	{
		////////////////////////////////////////////////
		// Pre-allocate GPU classes for self-reference /
		////////////////////////////////////////////////

		const struct MYRIADOBJECT_CLASS *obj_addr = NULL, *class_addr = NULL;
	
		//TODO: Not sure if we need these; surely we can just use object[x].size instead?
		const size_t obj_size = sizeof(struct MYRIADOBJECT_OBJECT);
		const size_t class_size = sizeof(struct MYRIADOBJECT_CLASS);

		// Allocate class and object structs on the GPU.
		CUDA_CHECK_RETURN(cudaMalloc((void**)&obj_addr, class_size)); 
		CUDA_CHECK_RETURN(cudaMalloc((void**)&class_addr, class_size));

		///////////////////////////////////////////////////
		// Static initialization using "Anonymous"  Class /
		///////////////////////////////////////////////////

		const struct MYRIADOBJECT_CLASS anon_class_class = {
			{class_addr}, // MyriadClass' class is itself
			obj_addr,     // Superclass is MyriadObject (a Class is an Object)
			class_addr,   // Device class is itself (since we're on the GPU)
			class_size,   // Size is the class size (methods and all)
			NULL,         // No constructor on the GPU
			NULL,         // No destructor on the GPU
			NULL,         // No cudafication; we're already on the GPU!
			NULL,         // No decudafication; we *stay* on the GPU.
		};

		CUDA_CHECK_RETURN(
			cudaMemcpy(
				(void**) class_addr,
				&anon_class_class,
				sizeof(struct MYRIADOBJECT_CLASS),
				cudaMemcpyHostToDevice
				)
			);	

		// Remember to update static CPU class object
		object[1].ONDEVICE_CLASS = class_addr; //TODO: Replace with memcpy?

		/////////////////////////////////////////////////////////
		// Static initialization using "Anonymous" Object Class /
		/////////////////////////////////////////////////////////
	
		const struct MYRIADOBJECT_CLASS anon_obj_class = {
			{class_addr}, // It's class is MyriadClass (on GPU, of course)
			obj_addr,     // Superclass is itself
			class_addr,   // Device class is it's class (since we're on the GPU)
			obj_size,     // Size is effectively a pointer
			NULL,         // No constructor on the GPU
			NULL,         // No destructor on the GPU
			NULL,         // No cudafication; we're already on the GPU!
			NULL,         // No decudafication; we *stay* on the GPU
		};
	
		CUDA_CHECK_RETURN(
			cudaMemcpy(
				(void**) obj_addr,
				&anon_obj_class,
				sizeof(struct MYRIADOBJECT_CLASS),
				cudaMemcpyHostToDevice
				)
			);
	
		// Remember to update static CPU object
		object[0].ONDEVICE_CLASS = (const struct MYRIADOBJECT_CLASS*) obj_addr; //TODO: Replace with memcpy?

		/////////////////////////////////////////////////
		// Memcpy GPU class pointers to *_dev_t symbols /
		/////////////////////////////////////////////////

		CUDA_CHECK_RETURN(
			cudaMemcpyToSymbol(
				(const void*) &MYRIAD_CAT(MYRIADOBJECT_CLASS, _dev_t), 
				&class_addr,
				sizeof(void*),
				0,
				cudaMemcpyHostToDevice
				)
			);

		CUDA_CHECK_RETURN(
			cudaMemcpyToSymbol(
				(const void*) &MYRIAD_CAT(MYRIADOBJECT_OBJECT, _dev_t),
				&obj_addr,
				sizeof(void*),
				0,
				cudaMemcpyHostToDevice
				)
			);

		return 0;
	} 
    #else
	{
		return EXIT_FAILURE;
	}
	#endif
}

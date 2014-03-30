#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>
#include <string.h>

#include "myriad_debug.h"

#include "MyriadObject.h"
#include "MyriadObject.cuh"


///////////////////////////////////////////////////////
// Static initalization for new()/classof() purposes //
///////////////////////////////////////////////////////
static void* MyriadObject_ctor(void* _self, va_list* app);
static void* MyriadClass_ctor(void* _self, va_list* app);
static void* MyriadObject_cudafy(void* self_obj, int clobber);
static void* MyriadClass_cudafy(void* _self, int clobber);

// Static, on-stack initialization of MyriadObject and MyriadClass classes
// Necessary because of circular dependencies (see comments below)
static struct MyriadClass object[] =
{
	// MyriadObject "anonymous" class
    {
        { object + 1 },              // MyriadClass is it's class
        object,                      // Superclass is itself (MyriadObject)
        NULL,                        // No device class by default
        sizeof(struct MyriadObject), // Size is effectively of pointer
        MyriadObject_ctor,           // Non-class constructor
		MyriadObject_cudafy,         // Gets on device as an object
    },
	// MyriadClass class
    {
        { object + 1 },             // MyriadClass is it's class
        object,                     // Superclass is MyriadObject (a Class is an Object)
        NULL,                       // No device class by default
        sizeof(struct MyriadClass), // Size includes methods, embedded MyriadObject
        MyriadClass_ctor,           // Constructor allows for prototype classes
		MyriadClass_cudafy,         // Cudafication to avoid static init for extensions
    }
};

const void* MyriadObject = object;
const void* MyriadClass = object + 1;

static void* MyriadObject_ctor(void* _self, va_list* app)
{
    return _self;
}

static void* MyriadObject_cudafy(void* self_obj, int clobber)
{
    struct MyriadObject* self = (struct MyriadObject*) self_obj;
	void* n_dev_obj = NULL;
	size_t my_size = myriad_size_of(self);

	const struct MyriadClass* tmp = self->m_class;
	self->m_class = self->m_class->device_class;

    CUDA_CHECK_RETURN(cudaMalloc(&n_dev_obj, my_size));

    CUDA_CHECK_RETURN(
		cudaMemcpy(
			n_dev_obj,
			self,
			my_size,
			cudaMemcpyHostToDevice
			)
		);

	self->m_class = tmp;

	return n_dev_obj;
}

//////////////////////////////////////////////
// MyriadClass-specific static methods //
//////////////////////////////////////////////

static void* MyriadClass_ctor(void* _self, va_list* app)
{
    struct MyriadClass* self = (struct MyriadClass*) _self;
    const size_t offset = offsetof(struct MyriadClass, my_ctor);

    self->super = va_arg(*app, struct MyriadClass*);
    self->size = va_arg(*app, size_t);

    assert(self->super);
	
	// Memcopies MyriadObject ctor and cudafy methods onto self (in case defaults aren't set)
    memcpy((char*) self + offset,
       (char*) self->super + offset,
       myriad_size_of(self->super) - offset);

    voidf selector;
    va_list ap;
    va_copy(ap, *app);

    while ((selector = va_arg(ap, voidf)))
    {
        voidf curr_method = va_arg(ap, voidf);
    
        if (selector == (voidf) myriad_ctor)
        {
            *(voidf *) &self->my_ctor = curr_method;
        } else if (selector == (voidf) myriad_cudafy) {
			*(voidf *) &self->my_cudafy = curr_method;
		}
    }

    return self;
}

// IMPORTANT: This is, ironically, for external classes' use only, since our 
// own initialization for MyriadClass is static and handled by initCUDAObjects
static void* MyriadClass_cudafy(void* _self, int clobber)
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

	struct MyriadClass* self = (struct MyriadClass*) _self;

	const struct MyriadClass* dev_class = NULL;
	const size_t class_size = myriad_size_of(self); // DO NOT USE sizeof(struct MyriadClass)!

	// Allocate space for new class on the card
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_class, class_size));
	
	// Memcpy the entirety of the old class onto a new CPU heap pointer
	const struct MyriadClass* class_cpy = (const struct MyriadClass*) calloc(1, class_size);
	memcpy((void*)class_cpy, _self, class_size);

	// Embedded object's class set to our GPU class; this is unaffected by $clobber
	memcpy((void*)&class_cpy->_.m_class, &dev_class, sizeof(void*)); 

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

/////////////////////////////////////
// Object management and Selectors //
/////////////////////////////////////

//----------------------------
//            New
//----------------------------

void* myriad_new(const void* _class, ...)
{
    const struct MyriadClass* prototype_class = (const struct MyriadClass*) _class;
    struct MyriadObject* curr_obj;
    va_list ap;

    assert(prototype_class && prototype_class->size);
    
    curr_obj = (struct MyriadObject*) calloc(1, prototype_class->size);
    assert(curr_obj);

    curr_obj->m_class = prototype_class;

    va_start(ap, _class);
    curr_obj = (struct MyriadObject*) myriad_ctor(curr_obj, &ap);
    va_end(ap);
	
    return curr_obj;
}

//----------------------------
//         Class Of
//----------------------------

const void* myriad_class_of(const void* _self)
{
    const struct MyriadObject* self = (const struct MyriadObject*) _self;
    return self->m_class;
}

//----------------------------
//         Size Of
//----------------------------

size_t myriad_size_of(const void* _self)
{
    const struct MyriadClass* m_class = (const struct MyriadClass*) myriad_class_of(_self);

    return m_class->size;
}

//----------------------------
//         Is A
//----------------------------

int myriad_is_a(const void* _self, const struct MyriadClass* m_class)
{
    return _self && myriad_class_of(_self) == m_class;
}

//----------------------------
//          Is Of
//----------------------------

int myriad_is_of(const void* _self, const struct MyriadClass* m_class)
{
    if (_self)
    {   
        const struct MyriadClass * myClass = (const struct MyriadClass*) myriad_class_of(_self);

        if (m_class != MyriadObject)
        {
            while (myClass != m_class)
            {
                if (myClass != MyriadObject)
                {
                    myClass = (const struct MyriadClass*) myriad_super(myClass);
                } else {
                    return 0;
                }
            }
        }

        return 1;
    }

    return 0;
}

//----------------------------
//    Constructor & CUDAFY
//----------------------------

void* myriad_ctor(void* _self, va_list* app)
{
    const struct MyriadClass* m_class = (const struct MyriadClass*) myriad_class_of(_self);

    assert(m_class->my_ctor);
    return m_class->my_ctor(_self, app);
}

void* myriad_cudafy(void* _self, int clobber)
{
    const struct MyriadClass* m_class = (const struct MyriadClass*) myriad_class_of(_self);

	assert(m_class->my_cudafy);
	return m_class->my_cudafy(_self, clobber);
}

///////////////////////////////
// Super and related methods //
///////////////////////////////

const void* myriad_super(const void* _self)
{
    const struct MyriadClass* self = (const struct MyriadClass*) _self;

    assert(self && self->super);
    return self->super;
}

void* super_ctor(const void* _class, void* _self, va_list* app)
{
    const struct MyriadClass* superclass = (const struct MyriadClass*) myriad_super(_class);

    assert(_self && superclass->my_ctor);
    return superclass->my_ctor(_self, app);
}

void* super_cudafy(const void* _class, void* _self, int clobber)
{
	const struct MyriadClass* superclass = (const struct MyriadClass*) myriad_super(_class);
	assert(_self && superclass->my_cudafy);
	return superclass->my_cudafy(_self, clobber);
}


///////////////////////////////////
//   CUDA Object Initialization  //
///////////////////////////////////

int initCUDAObjects()
{
	////////////////////////////////////////////////
	// Pre-allocate GPU classes for self-reference /
	////////////////////////////////////////////////

	const struct MyriadClass *obj_addr = NULL, *class_addr = NULL;
	
	//TODO: Not sure if we need these; surely we can just use object[x].size instead?
	const size_t obj_size = sizeof(struct MyriadObject);
	const size_t class_size = sizeof(struct MyriadClass);

	CUDA_CHECK_RETURN(cudaMalloc((void**)&obj_addr, class_size)); 
	CUDA_CHECK_RETURN(cudaMalloc((void**)&class_addr, class_size));

	///////////////////////////////////////////////////
	// Static initialization using "Anonymous"  Class /
	///////////////////////////////////////////////////

	const struct MyriadClass anon_class_class = {
		{class_addr}, // MyriadClass' class is itself
		obj_addr,     // Superclass is MyriadObject (a Class is an Object)
		class_addr,   // Device class is itself (since we're on the GPU)
		class_size,   // Size is the class size (methods and all)
		NULL,         // No constructor on the GPU
		NULL          // No cudafication; we're already on the GPU!
	};

	CUDA_CHECK_RETURN(
		cudaMemcpy(
			(void**) class_addr,
			&anon_class_class,
			sizeof(struct MyriadClass),
			cudaMemcpyHostToDevice
			)
		);	

	// Remember to update static CPU class object
	object[1].device_class = class_addr; //TODO: Replace with memcpy?

	/////////////////////////////////////////////////////////
	// Static initialization using "Anonymous" Object Class /
	/////////////////////////////////////////////////////////
	
	const struct MyriadClass anon_obj_class = {
		{class_addr}, // It's class is MyriadClass (on GPU, of course)
		obj_addr,     // Superclass is itself
		class_addr,   // Device class is it's class (since we're on the GPU)
		obj_size,     // Size is effectively a pointer
		NULL,         // No constructor on the GPU
		NULL,         // No cudafication; we're already on the GPU!
	};
	
	CUDA_CHECK_RETURN(
		cudaMemcpy(
			(void**) obj_addr,
			&anon_obj_class,
			sizeof(struct MyriadClass),
			cudaMemcpyHostToDevice
			)
		);
	
	// Remember to update static CPU object
	object[0].device_class = (const struct MyriadClass*) obj_addr; //TODO: Replace with memcpy?

	/////////////////////////////////////////////////
	// Memcpy GPU class pointers to *_dev_t symbols /
	/////////////////////////////////////////////////

	CUDA_CHECK_RETURN(
		cudaMemcpyToSymbol(
			(const void*) &MyriadClass_dev_t,
			&class_addr,
			sizeof(void*),
			0,
			cudaMemcpyHostToDevice
			)
		);

	CUDA_CHECK_RETURN(
		cudaMemcpyToSymbol(
			(const void*) &MyriadObject_dev_t,
			&obj_addr,
			sizeof(void*),
			0,
			cudaMemcpyHostToDevice
			)
		);

	return 0;
}

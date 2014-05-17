/**
   @file    MyriadObject.h
 
   @brief   Generic MyriadObject class definition file.
 
   @details Defines the Myriad object-oriented system base classes & functions
 
   @author  Pedro Rittner
 
   @date    April 7, 2014
 */
#ifndef MYRIADOBJECT_H
#define MYRIADOBJECT_H

#include <stddef.h>
#include <stdarg.h>

#include "MyriadObject_meta.h"

// TODO: Move to some utility library
#ifndef INDIRECT_SET
#define INDIRECT_SET(struct_ptr, struct_name, member_name, var, type)          \
    memcpy(                                                                    \
        (int*)((char*)struct_ptr + offsetof(struct struct_name, member_name)), \
        &var,                                                                  \
        sizeof(type))
#endif

////////////////////////////////
// Function pointer type defs //
////////////////////////////////

//! Generic function pointer type
typedef void (* voidf) ();

//! Constructor function pointer type
typedef void* (* ctor_t) (void* self, va_list* app);

//! Destructor function pointer type
typedef int (* dtor_t) (void* self);

//! CUDAfy function pointer type
typedef void* (* cudafy_t) (void* self, int clobber);

//! De-CUDAfy function pointer type
typedef void (* de_cudafy_t) (void* self, void* cuda_self);


/////////////////////////////////
// Struct forward declarations //
/////////////////////////////////

struct MyriadObject;
struct MyriadClass;

////////////////////////////////////////////////
// Dynamically-initialized reference pointers //
////////////////////////////////////////////////

extern const void* MyriadObject;   // new(MyriadObject); 
extern const void* MyriadClass;    // new(MyriadClass, super, ..., 0);

/**
   Initializes on-GPU CUDA prototype objects to support Myriad OOP on the GPU.
  
   Preallocates on-GPU dynamic prototype objects that will be used as reference
   for Myriad methods, such as myriad_class_of and other OOP features.
  
   @returns EXIT_SUCCESS if creation completed successfully, EXIT_FAILURE o.w.
 */
extern int initCUDAObjects();

/////////////////////////////////////
// Object management and Selectors //
/////////////////////////////////////

/**
   Creates a new object in memory, given its prototype class.
   
   Allocates the memory for the new object in memory and delegates the work to
   the prototype class constructor
 
   @param[in]    _class    prototype class object (e.g. MyriadObject)
 
   @returns pointer to the newly-created object, NULL if creation failed.
 */
extern void* myriad_new(const void* _class, ...);

/**
   Returns the reference class pointer of a given object instance.
   
   @param[in]    _self    pointer to extant object instance
   
   @returns pointer to the reference class of the instance object
 */
extern const void* myriad_class_of(const void* _self);

/**
   Returns the size of a given generic object.
   
   @param[in]    _self    pointer to extant generic object 

   @returns size of the given object
 */
extern size_t myriad_size_of(const void* self);

/**
   Tests if a given object is an instance of a given prototype class.
   
   @param[in]    _self      pointer to extant object instance
   @param[in]    m_class    pointer to extant prototype class
   
   @returns 1 if true, 0 if otherwise
*/
extern int myriad_is_a(const void* _self, const struct MyriadClass* m_class);

/**
   Tests if a given object instance inherets from a given prototype class.

   @param[in]    _self      pointer to extant object instance
   @param[in]    m_class    pointer to extant prototype class
   
   @returns 1 if true, 0 if otherwise
   
 */
extern int myriad_is_of(const void* _self, const struct MyriadClass* m_class);

///////////////////////////////////////
// Generic Object Function Delegates //
///////////////////////////////////////

/**
   Calls constructor for the given object.
 
   Note that for MyriadObject this is passthrough, but it's necessary for it to
   exit to have a reference point for the selector in the MyriadClass_ctor.
  
   @param[in]    _self  pointer to the object's memory as it currently exists
   @param[in]    app    variable arguments list from @see myriad_new

   @returns newly created object as a generic void pointer
 */
extern void* myriad_ctor(void* _self, va_list* app);

/**
   Calls destructor for the given object.

   Note that calling a destructor multiple times is undefined behavior.
   Note that calling a destructor on a MyriadClass is undefined behavior.

   @param[in]    _self    pointer to an extant object in memory
   
   @returns EXIT_SUCCESS if memory was successfully freed, EXIT_FAILURE o.w.
 */
extern int myriad_dtor(void* _self);

/**
   Calls CUDAfy method for the given object, with clobber flag.
   
   @param[in]    _self    pointer to an extant object in memory
   @param[in]    clobber  flag for overriding underclass' CUDAfy call

   @returns pointer in CUDA device memory to new object instance
 */
extern void* myriad_cudafy(void* _self, int clobber);

/**
   Calls deCUDAfy method for the given object.

   Note that this overrides the extant CPU object with the GPU contents.

   @param[in,out]  _self      pointer to an extant object in memory
   @param[in]      cuda_self  pointer to extant object in CUDA device memory 
 */
extern void myriad_decudafy(void* _self, void* cuda_self);


///////////////////////////////
// Super and related methods //
///////////////////////////////

/**
   Gets pointer to the super-class of the given object.

   @param[in]    _self    pointer to an extant object in memory
   
   @returns pointer to the superclass of the given object
 */
extern const void* myriad_super(const void* _self);

/**
   Calls the superclass' constructor for an underclass object.

   @param[in]    _class  pointer to the prototype superclass
   @param[in]    _self   pointer to an extant object in memory
   @param[in]    app     pointer to the ctor argument list

   @returns result of the superclass' constructor
   
   @see myriad_ctor
 */
extern void* super_ctor(const void* _class, void* _self, va_list* app);

/**
   Calls the superclass' destructor for an underclass object.

   @param[in]    _class  pointer to the prototype superclass
   @param[in]    _self   pointer to an extant object in memory

   @returns result of the superclass' destructor

   @see myriad_dtor
 */
extern int super_dtor(const void* _class, void* _self);

/**
   Calls the superclass' destructor for an underclass object.

   @param[in]    _class   pointer to the prototype superclass
   @param[in]    _self    pointer to an extant object in memory
   @param[in]    clobber  flag for overriding underclass' CUDAfy call

   @returns result of the superclass' CUDAfyer

   @see myriad_cudafy
 */
extern void* super_cudafy(const void* _class, void* _self, int clobber);

/**
   Calls the superclass' deCUDAfyer for an underclass object.

   @param[in]      _class     pointer to the prototype superclass
   @param[in,out]  _self      pointer to an extant object in memory
   @param[in]      cuda_self  pointer to extant object in CUDA device memory 

   Note that like myriad_decudafy this can override the extant object's contents

   @see myriad_decudafy
 */
extern void super_decudafy(const void* _class, void* _self, void* cuda_self);

////////////////////////////////////////
// Object & Class Struct declarations //
////////////////////////////////////////

/**
   Base object underpinning the Myriad OOP model.

   MyriadObjects are designed to store state and data.
   Any and all functionality is in the self-describing class prototype this
   object points to.
 */
struct MyriadObject
{
    const struct MyriadClass* m_class; //! Object's class/description
};

/**
   Base class underpinning the Myriad OOP model.

   Classes are single-instance reference objects to which all object instances
   point to. They define all internally-supported functionality.
 */
struct MyriadClass
{
    const struct MyriadObject _;               //! Embedded object
    const struct MyriadClass* super;           //! Super Class
    const struct MyriadClass* device_class;    //! On-device class
    size_t size;                               //! Object size
    ctor_t my_ctor;                            //! Constructor
	dtor_t my_dtor;                            //! Destructor
	cudafy_t my_cudafy;                        //! CUDAfier
	de_cudafy_t my_decudafy;                   //! De-CUDAficator
};

#endif

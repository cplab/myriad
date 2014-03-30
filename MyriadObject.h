#ifndef MYRIADOBJECT_H
#define MYRIADOBJECT_H

#include <stddef.h>
#include <stdarg.h>

// TODO: Move to some utility library
#ifndef INDIRECT_SET
#define INDIRECT_SET(struct_ptr, struct_name, member_name, var, type)           \
    memcpy(                                                                     \
        (int*)((char*)struct_ptr + offsetof(struct struct_name, member_name)),  \
        &var,                                                                   \
        sizeof(type))
#endif

////////////////////////////////
// Function pointer type defs //
////////////////////////////////

//! Generic function pointer type
typedef void (* voidf) ();
//! Constructor function pointer type
typedef void* (* ctor_t) (void* self, va_list* app);
//! Cudafy function pointer type
typedef void* (* cudafy_t) (void* self, int clobber);

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

extern int initCUDAObjects();

/////////////////////////////////////
// Object management and Selectors //
/////////////////////////////////////

extern void* myriad_new(const void* _class, ...);

extern const void* myriad_class_of (const void* _self);

extern size_t myriad_size_of(const void* self);

extern int myriad_is_a(const void* _self, const struct MyriadClass* m_class);

/** Calls constructor for the given class.
 *
 * Note that for MyriadObject and MyriadClass this is passthrough, but 
 * it's necessary for it to exit to have a reference point for the selector in 
 * the MyriadClass_ctor.
 * 
 * @param[in,out]  self_obj     
 * @param[in]  _self
 * @param[in]  app
 * @return
 */
extern void* myriad_ctor (void* self, va_list * app);

extern void* myriad_cudafy(void* _self, int clobber);

extern int myriad_is_of(
	const void* _self,
    const struct MyriadClass* m_class
);

///////////////////////////////
// Super and related methods //
///////////////////////////////

extern const void* myriad_super(const void* _self);

extern void* super_ctor(const void* _class, void* _self, va_list* app);

extern void* super_cudafy(const void* _class, void* _self, int clobber);


////////////////////////////////////////
// Object & Class Struct declarations //
////////////////////////////////////////

struct MyriadObject
{
    const struct MyriadClass* m_class; //! Object's class/description
};

struct MyriadClass
{
    const struct MyriadObject _;               //! Embedded object
    const struct MyriadClass* super;           //! Super Class
    const struct MyriadClass* device_class;    //! On-device class
    size_t size;                               //! Object size
    ctor_t my_ctor;                            //! Constructor
	cudafy_t my_cudafy;                        //! CUDA-fication
};

#endif

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "MyriadObject.cuh"

__constant__ __device__ struct MyriadClass* MyriadObject_dev_t = NULL;
__constant__ __device__ struct MyriadClass* MyriadClass_dev_t = NULL;

/////////////////////////////////////
// Object management and Selectors //
/////////////////////////////////////

//----------------------------
//         Class Of
//----------------------------

__device__ const void* cuda_myriad_class_of(const void* _self)
{
    const struct MyriadObject* self = (const struct MyriadObject*) _self;
    return self->m_class;
}

//----------------------------
//         Size Of
//----------------------------

__device__ size_t cuda_myriad_size_of(const void* _self)
{
	const struct MyriadClass* m_class = (const struct MyriadClass*) cuda_myriad_class_of(_self);
	return m_class->size;
}

//----------------------------
//         Is A
//----------------------------

__device__ int cuda_myriad_is_a(const void* _self, const struct MyriadClass* m_class)
{
	return _self && cuda_myriad_class_of(_self) == m_class;
}

//----------------------------
//          Is Of
//----------------------------

__device__ int cuda_myriad_is_of(const void* _self, const struct MyriadClass* m_class)
{
	if (_self)
	{
		const struct MyriadClass* myClass = (const struct MyriadClass*) cuda_myriad_class_of(_self); 
		
		if ((void*) m_class != MyriadObject_dev_t)
		{
			while (myClass != m_class)
			{
				if ((void*) myClass != MyriadObject_dev_t)
				{
					myClass = (const struct MyriadClass*) cuda_myriad_super(myClass);
				} else {
					return 0;
				}
			}
		}
		
		return 1;
	}

	return 0;
}

///////////////////////////////
// Super and related methods //
///////////////////////////////

__device__ const void* cuda_myriad_super(const void* _self)
{
	const struct MyriadClass* self = (const struct MyriadClass*) _self;

	return self->super;
}


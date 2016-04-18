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


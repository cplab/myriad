#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

// Myriad C API Headers
extern "C"
{
    #include "myriad_debug.h"
}

#include "MyriadObject.cuh"
#include "Mechanism.cuh"
#include "Compartment.cuh"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

/////////////////////////////////////
//! Tests basic CUDA functionality //
/////////////////////////////////////
__global__ void check_obj_kernel(void* cuda_obj)
{
    const struct MyriadObject* o = (const struct MyriadObject*) cuda_obj;
    printf("\tcuda_obj->m_class: %p\n", o->m_class);
}

static int cuda_basic_test()
{
    void* cuda_obj = myriad_new(MyriadObject);
	assert(cuda_obj);

    printf("\tMyriadObject: %p\n", MyriadObject);

    const int nThreads = 1; // NUM_CUDA_THREADS;
    const int nBlocks = 1;

    dim3 dimGrid(nBlocks);
    dim3 dimBlock(nThreads);

    // Test
    #ifndef __clang__
    check_obj_kernel<<<dimGrid, dimBlock>>>(cuda_obj); // Not an error
    #endif
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    cudaDeviceReset();

    return EXIT_SUCCESS;
}

//////////////////////////////////////////
// Test fancy get address functionality //
//////////////////////////////////////////

__device__ struct MyriadObject c_dev_obj = {NULL};
__device__ struct MyriadClass c_dev_class = {
            {NULL},
            NULL,
            0,
            NULL,
			NULL,
			NULL,
        };

__global__ void fancy_addr_check()
{
    printf("\tsize(GPU): %lu\n", c_dev_class.size);
}

static int cuda_address_test()
{
    void* addr_of_object = NULL, *addr_of_class = NULL;
    CUDA_CHECK_RETURN(cudaGetSymbolAddress(&addr_of_object, c_dev_obj));
    CUDA_CHECK_RETURN(cudaGetSymbolAddress(&addr_of_class, c_dev_class));

    struct MyriadClass c_class = {
        {(const struct MyriadClass*) addr_of_class},
        (const struct MyriadClass*) addr_of_class,
        (const struct MyriadClass*) addr_of_class,
        sizeof(struct MyriadClass),
		NULL,
		NULL,
    };

    printf("\tsize(CPU): %lu\n",c_class.size);

    CUDA_CHECK_RETURN(
        cudaMemcpyToSymbol(
            c_dev_class,
            &c_class,
            sizeof(struct MyriadClass),
            0,
            cudaMemcpyHostToDevice
        )
    );

    const int nThreads = 1; // NUM_CUDA_THREADS;
    const int nBlocks = 1;

    dim3 dimGrid(nBlocks);
    dim3 dimBlock(nThreads);

    // Test
    #ifndef __clang__
    fancy_addr_check<<<dimGrid, dimBlock>>>(); // Not an error
    #endif
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    cudaDeviceReset();

    return EXIT_SUCCESS;
}

///////////////////
// Test CUDA OOP //
///////////////////

__global__ void cuda_oop_test(void* c_obj)
{
    printf("\tsize(GPU): %lu\n", cuda_myriad_size_of(c_obj));
    printf("\tis_a: %s\n", cuda_myriad_is_a(c_obj, MyriadObject_dev_t) ? "TRUE" : "FALSE");
    printf("\tis_of: %s\n", cuda_myriad_is_of(c_obj, MyriadObject_dev_t) ? "TRUE" : "FALSE");
}

static int cuda_oop()
{
    initCUDAObjects();
    
    void* my_obj = myriad_new(MyriadObject);
	assert(my_obj);
    
    void* my_cuda_obj = myriad_cudafy(my_obj, 0);

    // BLAH
    const int nThreads = 1; // NUM_CUDA_THREADS;
    const int nBlocks = 1;

    dim3 dimGrid(nBlocks);
    dim3 dimBlock(nThreads);

    // Test
    #ifndef __clang__
    cuda_oop_test<<<dimGrid, dimBlock>>>(my_cuda_obj); // Not an error
    #endif
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    cudaDeviceReset();

    printf("\tCPU Size (again): %lu\n", myriad_size_of(my_obj));

    return EXIT_SUCCESS;
}

////////////////////
// Test Mechanism //
////////////////////

__global__ void cuda_mechansim_test(void* obj)
{
	struct Mechanism* self = (struct Mechanism*) obj;
	struct MechanismClass* self_c = (struct MechanismClass*) cuda_myriad_class_of(self);
	printf("\tMy ptr: %p\n", self);
	printf("\tMy ID: %i\n", self->source_id);
	printf("\tMy class: %p\n", self->_.m_class);
	printf("\tGPU, my size: %lu\n", cuda_myriad_size_of(obj));
	printf("\tMechanism fxn: %p\n", self_c->m_mech_fxn);
	printf("\tMechanism fxn invocation: %f\n", self_c->m_mech_fxn(self, NULL, NULL, 0.0, 0.0, 0));
	printf("\tMechanism fxn indirect call: %f\n", cuda_mechanism_fxn(self, NULL, NULL, 0.0, 0.0, 0));
}

static int mechanism_test()
{
	initCUDAObjects();
	initMechanism(1);

	void* mech_obj = NULL, *dev_mech_obj = NULL;

	mech_obj = myriad_new(Mechanism, 1);

	UNIT_TEST_VAL_EQ(myriad_size_of(mech_obj), sizeof(struct Mechanism));

	mechanism_fxn(mech_obj, NULL, NULL, 0, 0, 0);

	dev_mech_obj = myriad_cudafy(mech_obj, 0);

    // BLAH
    const int nThreads = 1; // NUM_CUDA_THREADS;
    const int nBlocks = 1;

    dim3 dimGrid(nBlocks);
    dim3 dimBlock(nThreads);

    // Test
    #ifndef __clang__
    cuda_mechansim_test<<<dimGrid, dimBlock>>>(dev_mech_obj); // Not an error
    #endif
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    cudaDeviceReset();

    return EXIT_SUCCESS;
}

//////////////////////
// Test Compartment //
//////////////////////

__global__ void cuda_compartment_test(void* obj)
{
	struct Compartment* self = (struct Compartment*) obj;
	struct CompartmentClass* self_c = (struct CompartmentClass*) cuda_myriad_class_of(self);
	printf("\tMy ptr: %p\n", self);
	printf("\tMy ID: %i\n", self->id);
	printf("\tMy class: %p\n", self->_.m_class);
	printf("\tGPU, my size: %lu\n", cuda_myriad_size_of(obj));
	printf("\tCompartment fxn: %p\n", self_c->m_comp_fxn);
	printf("\tCompartment fxn invocation: "); self_c->m_comp_fxn(self, NULL, 0.0, 0.0, 0);
	printf("\tCompartent fxn indirect call: "); cuda_simul_fxn(self, NULL, 0.0, 0.0, 0);
}

static int compartment_test()
{
	initCUDAObjects();
	initCompartment(1);

	void* comp_obj = NULL, *dev_comp_obj = NULL;

	comp_obj = myriad_new(Compartment, 1);

	UNIT_TEST_VAL_EQ(myriad_size_of(comp_obj), sizeof(struct Compartment));

	simul_fxn(comp_obj, NULL, 0.0, 0.0, 0);

	dev_comp_obj = myriad_cudafy(comp_obj, 0);

    // BLAH
    const int nThreads = 1; // NUM_CUDA_THREADS;
    const int nBlocks = 1;

    dim3 dimGrid(nBlocks);
    dim3 dimBlock(nThreads);

    // Test
    #ifndef __clang__
    cuda_compartment_test<<<dimGrid, dimBlock>>>(dev_comp_obj); // Not an error
    #endif
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    cudaDeviceReset();

	return EXIT_SUCCESS;
}

//////////////////////////////
// Test Device Symbol Malloc /
//////////////////////////////

__device__ float* my_float_ptr = NULL;

__global__ void cuda_dev_malloc_test()
{
	printf("my_float_ptr: %f\n", my_float_ptr[0]);
}

static int cuda_symbol_malloc()
{
	float* host_float_ptr = NULL, host_float_val = 5.0;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&host_float_ptr, sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(host_float_ptr, &host_float_val, sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(my_float_ptr, &host_float_ptr, sizeof(float*), size_t(0), cudaMemcpyHostToDevice));
	
	    
    // BLAH
    const int nThreads = 1; // NUM_CUDA_THREADS;
    const int nBlocks = 1;

    dim3 dimGrid(nBlocks);
    dim3 dimBlock(nThreads);

    // Test
    #ifndef __clang__
    cuda_dev_malloc_test<<<dimGrid, dimBlock>>>(); // Not an error
    #endif
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    CUDA_CHECK_RETURN(cudaGetLastError());

    cudaDeviceReset();

    return EXIT_SUCCESS;
}

///////////////////
// Main function //
///////////////////
int main(int argc, char const *argv[])
{
    puts("Hello World!\n");

    // UNIT_TEST_FUN(initCUDAObjects);
	// UNIT_TEST_FUN(cuda_basic_test);
    UNIT_TEST_FUN(cuda_address_test);
    UNIT_TEST_FUN(cuda_oop);
	UNIT_TEST_FUN(cuda_symbol_malloc);
	UNIT_TEST_FUN(mechanism_test);
	UNIT_TEST_FUN(compartment_test);

    puts("\nDone.");

    return EXIT_SUCCESS;
}

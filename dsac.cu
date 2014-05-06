#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#ifdef CUDA
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

// Myriad C API Headers
extern "C"
{
	#include "myriad_debug.h"
    #include "MyriadObject.h"
	#include "Mechanism.h"
	#include "Compartment.h"
	#include "HHSomaCompartment.h"
	#include "HHLeakMechanism.h"
	#include "HHNaCurrMechanism.h"
	#include "HHKCurrMechanism.h"
    #include "DCCurrentMech.h"
}

////////////////
// DSAC Model //
////////////////

// Simulation parameters
#define SIMUL_LEN 1000000 
#define DT 0.001
// Leak params
#define G_LEAK 1.0
#define E_REV -65.0
// Sodium params
#define G_NA 35.0
#define E_NA 55.0
#define HH_M 0.5
#define HH_H 0.1
// Potassium params
#define G_K 9.0
#define E_K -90.0
#define HH_N 0.1
// Compartment Params
#define CM 1.0
#define INIT_VM -65.0


#ifdef CUDA
__global__ void cuda_hh_compartment_test(void* hh_comp_obj, void* network)
{
	void* dev_arr[1];
	dev_arr[0] = network;

	struct HHSomaCompartment* curr_comp = (struct HHSomaCompartment*) hh_comp_obj;

	double curr_time = DT;
	for (unsigned int curr_step = 1; curr_step < SIMUL_LEN; curr_step++)
	{
		cuda_simul_fxn(curr_comp, (void**) dev_arr, DT, curr_time, curr_step);
		curr_time += DT;
	}
}
#endif

static int HHCompartmentTest()
{
	int cuda_init;

	#ifdef CUDA
    cuda_init = 1;
	initCUDAObjects();
	#else
   	cuda_init = 0;
	#endif

	initMechanism(cuda_init);
	initDCCurrMech(cuda_init);
	initHHLeakMechanism(cuda_init);
	initHHNaCurrMechanism(cuda_init);
	initHHKCurrMechanism(cuda_init);
	initCompartment(cuda_init);
	initHHSomaCompartment(cuda_init);

	void** network = (void**) calloc(1, sizeof(void*));

	void* hh_comp_obj = myriad_new(HHSomaCompartment, 0, 0, NULL, SIMUL_LEN, NULL, INIT_VM, CM);
	void* hh_leak_mech = myriad_new(HHLeakMechanism, 0, G_LEAK, E_REV);
	void* hh_na_curr_mech = myriad_new(HHNaCurrMechanism, 0, G_NA, E_NA, HH_M, HH_H);
	void* hh_k_curr_mech = myriad_new(HHKCurrMechanism, 0, G_K, E_K, HH_N);
	void* dc_curr_mech = myriad_new(DCCurrentMech, 0, 200000, 999000, 9.0);

	#ifdef CUDA
	{
		void* cuda_na_mech = myriad_cudafy(hh_na_curr_mech, 0);
		void* cuda_leak_mech = myriad_cudafy(hh_leak_mech, 0);
		void* cuda_k_mech = myriad_cudafy(hh_k_curr_mech, 0);
		void* cuda_dc_mech = myriad_cudafy(dc_curr_mech, 0);

		// Add mechanism to compartment
		assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, cuda_leak_mech));
		assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, cuda_na_mech));
		assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, cuda_k_mech));
		assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, cuda_dc_mech));

		void* cuda_comp_obj = myriad_cudafy(hh_comp_obj, 0);

		network[0] = cuda_comp_obj;

        const int nThreads = 1; // NUM_CUDA_THREADS;
        const int nBlocks = 1;

        dim3 dimGrid(nBlocks);
        dim3 dimBlock(nThreads);

        cuda_hh_compartment_test<<<dimGrid, dimBlock>>>(cuda_comp_obj, network[0]);
	    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        CUDA_CHECK_RETURN(cudaGetLastError());

		// Decudafy stuff
		myriad_decudafy(hh_comp_obj, cuda_comp_obj);
        cudaDeviceReset();
	}
	#else
	{
		assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, hh_leak_mech));
		assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, hh_na_curr_mech));
		assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, hh_k_curr_mech));
		assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, dc_curr_mech));

     	network[0] = hh_comp_obj;
		double curr_time = DT;
		for (unsigned int curr_step = 1; curr_step < SIMUL_LEN; curr_step++)
		{
	        simul_fxn(hh_comp_obj, network, DT, curr_time, curr_step);
        	curr_time += DT;
        }
    }
	#endif

	struct HHSomaCompartment* curr_comp = (struct HHSomaCompartment*) hh_comp_obj;
	FILE* p_file = fopen("output.dat","wb");
	fwrite(curr_comp->soma_vm, sizeof(double), curr_comp->soma_vm_len, p_file);
	fclose(p_file);

	// Free
	assert(EXIT_SUCCESS == myriad_dtor(hh_leak_mech));
	assert(EXIT_SUCCESS == myriad_dtor(hh_na_curr_mech));
	assert(EXIT_SUCCESS == myriad_dtor(hh_k_curr_mech));
	assert(EXIT_SUCCESS == myriad_dtor(hh_comp_obj));

    return EXIT_SUCCESS;
}

///////////////////
// Main function //
///////////////////
int main(int argc, char const *argv[])
{
    puts("Hello World!\n");

	UNIT_TEST_FUN(HHCompartmentTest);

    puts("\nDone.");

    return EXIT_SUCCESS;
}

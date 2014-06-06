#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

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
//    #include "HHGradedGABAAMechanism.h"
    #include "HHSpikeGABAAMechanism.h"
    #include "DCCurrentMech.h"
}

#ifdef CUDA
#include "MyriadObject.cuh"
#include "Mechanism.cuh"
#include "Compartment.cuh"
#endif


////////////////
// DSAC Model //
////////////////

// Simulation parameters
#define SIMUL_LEN 1000000 
#define DT 0.001
#define NUM_CELLS 2
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
// GABA-a Params
#define GABA_VM_THRESH 0.0
#define GABA_G_MAX 0.1
#define GABA_TAU_ALPHA 0.08333333333333333
#define GABA_TAU_BETA 10.0
#define GABA_REV -75.0

#ifdef CUDA

__global__ void myriad_simul(void** network)
{
	__shared__ double curr_time;
    __shared__ unsigned int curr_step;

    if (threadIdx.x == 1)
    {
        curr_time = DT;
        curr_step = 1;
    }
    __syncthreads();

	while (curr_step < SIMUL_LEN)
	{
		cuda_simul_fxn(network[threadIdx.x], network, DT, curr_time, curr_step);
        if (threadIdx.x == 1)
        {
            curr_time += DT;
            curr_step++;
        }
        __syncthreads();
	}
}
#endif

static void* new_dsac_soma(unsigned int id, unsigned int* connect_to, const unsigned int num_connxs)
{
	void* hh_comp_obj = myriad_new(HHSomaCompartment, id, 0, NULL, SIMUL_LEN, NULL, INIT_VM, CM);
	void* hh_leak_mech = myriad_new(HHLeakMechanism, id, G_LEAK, E_REV);
	void* hh_na_curr_mech = myriad_new(HHNaCurrMechanism, id, G_NA, E_NA, HH_M, HH_H);
	void* hh_k_curr_mech = myriad_new(HHKCurrMechanism, id, G_K, E_K, HH_N);

	void* dc_curr_mech = NULL;
	if (id == 0)
	{
		dc_curr_mech = myriad_new(DCCurrentMech, id, 200000, 999000, 9.0);
	} else {
		dc_curr_mech = myriad_new(DCCurrentMech, id, 200000, 999000, 0.0);
	}

    #ifdef CUDA
	assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, myriad_cudafy(hh_leak_mech, 0)));
	assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, myriad_cudafy(hh_na_curr_mech, 0)));
	assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, myriad_cudafy(hh_k_curr_mech, 0)));
	assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, myriad_cudafy(dc_curr_mech, 0)));
    #else
	assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, hh_leak_mech));
	assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, hh_na_curr_mech));
	assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, hh_k_curr_mech));
	assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, dc_curr_mech));
    #endif

	if (num_connxs > 0)
	{
		for (unsigned int i = 0; i < num_connxs; i++)
		{
			void* hh_GABA_a_curr_mech = 
				myriad_new
				(
					HHSpikeGABAAMechanism,
					connect_to[i],
                    GABA_VM_THRESH,
                    -INFINITY,
                    GABA_G_MAX,
                    GABA_TAU_ALPHA,
                    GABA_TAU_BETA,
                    GABA_REV
				);

            #ifdef CUDA
			assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, myriad_cudafy(hh_GABA_a_curr_mech, 0)));
            #else
			assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, hh_GABA_a_curr_mech));
            #endif

			printf("Made GABA synapse starting at cell %i ending at cell %i\n", connect_to[i], id);
		}
	}

    #ifdef CUDA
	return myriad_cudafy(hh_comp_obj, 0);
    #else
    return hh_comp_obj;
    #endif
}

static int dsac()
{
    #ifdef CUDA
	const int cuda_init = 1;
    #else
	const int cuda_init = 0;
    #endif

	initMechanism(cuda_init);
	initDCCurrMech(cuda_init);
	initHHLeakMechanism(cuda_init);
	initHHNaCurrMechanism(cuda_init);
	initHHKCurrMechanism(cuda_init);
	initHHSpikeGABAAMechanism(cuda_init);
	initCompartment(cuda_init);
	initHHSomaCompartment(cuda_init);

	void** network = NULL;

    #ifdef CUDA
    CUDA_CHECK_RETURN(cudaMallocHost(&network, NUM_CELLS * sizeof(void*)));
    #else
    network = (void**) calloc(NUM_CELLS, sizeof(void*));
    #endif

	for (int i = 0; i < NUM_CELLS; i++)
	{
		//TODO: Guarantee % connectivity b/w cells in network
		const unsigned int num_connxs = 1;
		unsigned int* to_connect = (unsigned int*) calloc(num_connxs, sizeof(unsigned int));
		
		//TODO: Get rid of this hack
		if (i == 0)
		{
			to_connect[0] = 1;
		} else if (i == 1) {
			to_connect[0] = 0;
		}

	    network[i] = new_dsac_soma(i, to_connect, 1);
        free(to_connect);
	}

    #ifdef CUDA

    void** c_network = NULL;
    CUDA_CHECK_RETURN(cudaMalloc(&c_network, sizeof(void*) * NUM_CELLS));
    CUDA_CHECK_RETURN(
        cudaMemcpy(
            c_network,
            network,
            sizeof(void*) * NUM_CELLS,
            cudaMemcpyHostToDevice
            )
        );

    const int nThreads = 2; // NUM_CUDA_THREADS;
    const int nBlocks = 1;

    dim3 dimGrid(nBlocks);
    dim3 dimBlock(nThreads);

    myriad_simul<<<dimGrid, dimBlock>>>(c_network);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    #else

	double curr_time = DT;
	for (unsigned int curr_step = 1; curr_step < SIMUL_LEN; curr_step++)
	{
		for (int i = 0; i < NUM_CELLS; i++)
		{
			simul_fxn(network[i], network, DT, curr_time, curr_step);
		}
		curr_time += DT;
	}

	for (int i = 0; i < NUM_CELLS; i++)
	{
		struct HHSomaCompartment* curr_comp = (struct HHSomaCompartment*) network[i];
		char* fname = (char*) calloc(1, sizeof("cell000.dat"));
		sprintf(fname, "cell%03i.dat", i);
		FILE* p_file = fopen(fname,"wb");
		fwrite(curr_comp->soma_vm, sizeof(double), curr_comp->soma_vm_len, p_file);
		fclose(p_file);
        free(fname);
	}
    #endif

    #ifdef CUDA
    CUDA_CHECK_RETURN(cudaFreeHost(network));
    #else
    free(network);
    #endif

    return EXIT_SUCCESS;
}

///////////////////
// Main function //
///////////////////
int main(int argc, char const *argv[])
{
    puts("Hello World!\n");

	assert(EXIT_SUCCESS == dsac());

    puts("\nDone.");

    return EXIT_SUCCESS;
}

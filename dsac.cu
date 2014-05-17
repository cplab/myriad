#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#ifdef CUDA
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MyriadObject.cuh"
#include "Mechanism.cuh"
#include "Compartment.cuh"
#include "HHSomaCompartment.cuh"
#include "HHLeakMechanism.cuh"
#include "HHNaCurrMechanism.cuh"
#include "HHKCurrMechanism.cuh"
#include "HHGradedGABAAMechanism.cuh"
#include "DCCurrentMech.cuh"
#endif

// Myriad C API Headers
#ifdef CUDA
extern "C"
{
#endif
	#include "myriad_debug.h"
    #include "MyriadObject.h"
	#include "Mechanism.h"
	#include "Compartment.h"
	#include "HHSomaCompartment.h"
	#include "HHLeakMechanism.h"
	#include "HHNaCurrMechanism.h"
	#include "HHKCurrMechanism.h"
	#include "HHGradedGABAAMechanism.h"
    #include "DCCurrentMech.h"
#ifdef CUDA
}
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
#define GABA_INIT 0.2
#define GABA_G_MAX 0.1
#define GABA_THETA 0.0			
#define GABA_SIGMA 2.0			
#define GABA_TAU_ALPHA 0.08333333333333333
#define GABA_TAU_BETA 10.0
#define GABA_REV -75.0

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

	assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, hh_leak_mech));
	assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, hh_na_curr_mech));
	assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, hh_k_curr_mech));
	assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, dc_curr_mech));

	if (num_connxs > 0)
	{
		for (unsigned int i = 0; i < num_connxs; i++)
		{
			void* hh_GABA_a_curr_mech = 
				myriad_new
				(
					HHGradedGABAAMechanism,
					connect_to[i],
					GABA_INIT,
					NULL,
					SIMUL_LEN,
					GABA_G_MAX,
					GABA_THETA,
					GABA_SIGMA,
					GABA_TAU_ALPHA,
					GABA_TAU_BETA,
					GABA_REV
				);
			assert(EXIT_SUCCESS == add_mechanism(hh_comp_obj, hh_GABA_a_curr_mech));
			printf("Made GABA synapse starting at cell %i ending at cell %i\n", connect_to[i], id);
		}
	}

	return hh_comp_obj;
}

static int dsac()
{
	const int cuda_init = 0;

	initMechanism(cuda_init);
	initDCCurrMech(cuda_init);
	initHHLeakMechanism(cuda_init);
	initHHNaCurrMechanism(cuda_init);
	initHHKCurrMechanism(cuda_init);
	initHHGradedGABAAMechanism(cuda_init);
	initCompartment(cuda_init);
	initHHSomaCompartment(cuda_init);

	void** network = (void**) calloc(NUM_CELLS, sizeof(void*));

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
	}	

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
		char* fname = (char*) malloc(sizeof("cell0.dat"));
		sprintf(fname, "cell%i.dat", i);
		FILE* p_file = fopen(fname,"wb");
		fwrite(curr_comp->soma_vm, sizeof(double), curr_comp->soma_vm_len, p_file);
		fclose(p_file);
	}

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

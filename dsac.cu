#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdbool.h>
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
#define NUM_CELLS 20
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

static void* new_dsac_soma(unsigned int id,
                           int64_t* connect_to,
                           bool stimulate,
                           const unsigned int num_connxs)
{
	void* hh_comp_obj = myriad_new(HHSomaCompartment, id, 0, NULL, SIMUL_LEN, NULL, INIT_VM, CM);
	void* hh_leak_mech = myriad_new(HHLeakMechanism, id, G_LEAK, E_REV);
	void* hh_na_curr_mech = myriad_new(HHNaCurrMechanism, id, G_NA, E_NA, HH_M, HH_H);
	void* hh_k_curr_mech = myriad_new(HHKCurrMechanism, id, G_K, E_K, HH_N);

	void* dc_curr_mech = NULL;
	if (stimulate)
	{
		dc_curr_mech = myriad_new(DCCurrentMech, id, 200000, 999000, 9.0);
	} else {
		dc_curr_mech = myriad_new(DCCurrentMech, id, 200000, 999000, 0.0);
	}

	assert(0 == add_mechanism(hh_comp_obj, hh_leak_mech));
	assert(0 == add_mechanism(hh_comp_obj, hh_na_curr_mech));
	assert(0 == add_mechanism(hh_comp_obj, hh_k_curr_mech));
	assert(0 == add_mechanism(hh_comp_obj, dc_curr_mech));

    for (uint64_t i = 0; i < num_connxs; i++)
    {
        // Don't connect if it's -1
        if (connect_to[i] == -1)
        {
            continue;
        }
            
        void* hh_GABA_a_curr_mech = myriad_new(HHSpikeGABAAMechanism,
                                               connect_to[i],
                                               GABA_VM_THRESH,
                                               -INFINITY,
                                               GABA_G_MAX,
                                               GABA_TAU_ALPHA,
                                               GABA_TAU_BETA,
                                               GABA_REV);
        assert(0 == add_mechanism(hh_comp_obj, hh_GABA_a_curr_mech));
        printf("GABA synapse from ID# %" PRIi64 " -> #ID %i\n",
               connect_to[i],
               id);
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
	initHHSpikeGABAAMechanism(cuda_init);
	initCompartment(cuda_init);
	initHHSomaCompartment(cuda_init);

	void* network[NUM_CELLS];
    // memset(network, 0, sizeof(void*) * NUM_CELLS);  // Necessary?
    
    const unsigned int num_connxs = NUM_CELLS;
    int64_t to_connect[num_connxs];

	for (unsigned int my_id = 0; my_id < NUM_CELLS; my_id++)
	{
        memset(to_connect, 0, sizeof(int64_t) * num_connxs);
        
		// All-to-All
        for (int64_t j = 0; j < NUM_CELLS; j++)
        {
            if (j == my_id)
            {
                to_connect[j] = -1;  // Don't connect to ourselves
            } else {
                to_connect[j] = j;   // Connect to cell j
            }
            printf("to_connect[%" PRIi64 "]: %" PRIi64 "\n", j, to_connect[j]);
        }
        
        const bool stimulate = rand() % 2 == 0;
	    network[my_id] = new_dsac_soma(my_id,
                                       to_connect,
                                       stimulate,
                                       num_connxs);
	}

    // Run simulation
	double curr_time = DT;
	for (unsigned int curr_step = 1; curr_step < SIMUL_LEN; curr_step++)
	{
		for (int i = 0; i < NUM_CELLS; i++)
		{
			simul_fxn(network[i], network, DT, curr_time, curr_step);
		}
		curr_time += DT;
	}

    return 0;
}

///////////////////
// Main function //
///////////////////
int main(int argc, char const *argv[])
{
    srand(42);
    puts("Hello World!\n");

	assert(0 == dsac());

    puts("\nDone.");

    return EXIT_SUCCESS;
}

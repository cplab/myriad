#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <unistd.h>

#ifdef CUDA
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

// Myriad C API Headers
extern "C"
{
    #include "myriad.h"
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
	void* hh_comp_obj = myriad_new(HHSomaCompartment, id, 0, NULL, NULL, INIT_VM, CM);
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

static ssize_t calc_total_size(int* num_allocs)
{
    ssize_t total_size = 0;
    
    // Scan module (read: current) directory for all C files
    /*
    int num_modules = 0;
    DIR* dp = opendir("./");  //TODO: Change this to look in module directory
    if (dp != NULL)
    {
        struct dirent *ep;
        while ((ep = readdir(dp)))
        {
            // Check if file is a C file
            if (strstr(ep->d_name, ".c") != NULL)
            {
                num_modules++;
            }
        }
        assert(closedir(dp) == 0);
    } else {
        perror("calc_total_size: Couldn't open the current directory");
        return -1;
    }
    */

    // Class Overhead
    total_size += sizeof(struct MyriadObject) + sizeof(struct MyriadClass);
    total_size += sizeof(struct Mechanism) + sizeof(struct MechanismClass);
    total_size += sizeof(struct DCCurrentMech) + sizeof(struct DCCurrentMechClass);
    total_size += sizeof(struct HHLeakMechanism) + sizeof(struct HHLeakMechanismClass);
    total_size += sizeof(struct HHNaCurrMechanism) + sizeof(struct HHNaCurrMechanismClass);
    total_size += sizeof(struct HHKCurrMechanism) + sizeof(struct HHKCurrMechanismClass);
    total_size += sizeof(struct HHSpikeGABAAMechanism) + sizeof(struct HHSpikeGABAAMechanismClass);
    total_size += sizeof(struct Compartment) + sizeof(struct CompartmentClass);
    total_size += sizeof(struct HHSomaCompartment) + sizeof(struct HHSomaCompartmentClass);
    *num_allocs = *num_allocs + (9 * 2);

    // Actual usage
    total_size += sizeof(struct HHSomaCompartment) * NUM_CELLS;
    total_size += sizeof(struct DCCurrentMech) * NUM_CELLS;
    total_size += sizeof(struct HHLeakMechanism) * NUM_CELLS;
    total_size += sizeof(struct HHNaCurrMechanism) * NUM_CELLS;
    total_size += sizeof(struct HHKCurrMechanism) * NUM_CELLS;
    total_size += sizeof(struct HHSpikeGABAAMechanism) * NUM_CELLS * NUM_CELLS;
    *num_allocs = *num_allocs + (6 * NUM_CELLS) + (NUM_CELLS * NUM_CELLS);
    
    return total_size;
}


static int dsac()
{
    #ifdef MYRIAD_ALLOCATOR
    int num_allocs = 0;
    const size_t total_mem_usage = calc_total_size(&num_allocs);
    assert(myriad_alloc_init(total_mem_usage, num_allocs) == 0);
    printf("total size: %lu, num allocs: %i\n", total_mem_usage, num_allocs);
    #endif

	initMechanism(false);
	initDCCurrMech(false);
	initHHLeakMechanism(false);
	initHHNaCurrMechanism(false);
	initHHKCurrMechanism(false);
	initHHSpikeGABAAMechanism(false);
	initCompartment(false);
	initHHSomaCompartment(false);

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
#pragma GCC ivdep
		for (int i = 0; i < NUM_CELLS; i++)
		{
			simul_fxn(network[i], network, DT, curr_time, curr_step);
		}
		curr_time += DT;
	}

    // Cleanup
    /* 
- Refactored all instances of "unsigned int" to uint64_t
- Changed cuda_init for initializers to use bool instead of int
- Added initial support for native myriad allocator
     */
    assert(myriad_finalize() == 0);

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

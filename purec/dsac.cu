#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <unistd.h>
#include <pthread.h>

#ifdef CUDA
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
const bool USE_CUDA = true;
#else
const bool USE_CUDA = false;
#endif

// Myriad C API Headers
#ifdef __cplusplus
extern "C" {
#endif

#include "myriad.h"
#ifdef MYRIAD_ALLOCATOR    
#include "myriad_alloc.h"
#endif
#include "MyriadObject.h"
#include "Mechanism.h"
#include "Compartment.h"
#include "HHSomaCompartment.h"
#include "HHLeakMechanism.h"
#include "HHNaCurrMechanism.h"
#include "HHKCurrMechanism.h"
#include "HHSpikeGABAAMechanism.h"
#include "DCCurrentMech.h"
    
#ifdef __cplusplus
}
#endif

#ifdef CUDA
#include "MyriadObject.cuh"
#include "Mechanism.cuh"
#include "Compartment.cuh"
#include "HHSomaCompartment.cuh"
#include "HHLeakMechanism.cuh"
#include "HHNaCurrMechanism.cuh"
#include "HHKCurrMechanism.cuh"
#include "HHSpikeGABAAMechanism.cuh"
#include "DCCurrentMech.cuh"
#endif

////////////////
// DSAC Model //
////////////////

// Fast exponential function structure/function
#ifdef FAST_EXP
__thread union _eco _eco;
#endif

static void* new_dsac_soma(unsigned int id,
                           int_fast32_t* connect_to,
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

	const int result =
        add_mechanism(hh_comp_obj, hh_leak_mech) ||
        add_mechanism(hh_comp_obj, hh_na_curr_mech) ||
        add_mechanism(hh_comp_obj, hh_k_curr_mech) ||
        add_mechanism(hh_comp_obj, dc_curr_mech);
    if (result)
    {
        fputs("Failed to add mechanisms to compartment", stderr);
        exit(EXIT_FAILURE);
    }

    for (uint_fast32_t i = 0; i < num_connxs; i++)
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
        if (add_mechanism(hh_comp_obj, hh_GABA_a_curr_mech))
        {
            fputs("Unable to add GABA current mechanism", stderr);
            exit(EXIT_FAILURE);
        }
        DEBUG_PRINTF("GABA synapse from ID# %li -> #ID %i\n", connect_to[i], id);
    }

	return hh_comp_obj;
}

static ssize_t calc_total_size(int* num_allocs)
{
    ssize_t total_size = 0;
    
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

    // Objects
    total_size += sizeof(struct HHSomaCompartment) * NUM_CELLS;
    total_size += sizeof(struct DCCurrentMech) * NUM_CELLS;
    total_size += sizeof(struct HHLeakMechanism) * NUM_CELLS;
    total_size += sizeof(struct HHNaCurrMechanism) * NUM_CELLS;
    total_size += sizeof(struct HHKCurrMechanism) * NUM_CELLS;
    total_size += sizeof(struct HHSpikeGABAAMechanism) * NUM_CELLS * NUM_CELLS;
    *num_allocs = *num_allocs + (6 * NUM_CELLS) + (NUM_CELLS * NUM_CELLS);

    return total_size;
}

#if NUM_THREADS > 1

static struct _pthread_vals
{
    void** network;
    double curr_time;
    uint_fast32_t curr_step;
    uint_fast32_t num_done;
    pthread_mutex_t barrier_mutx;
    pthread_cond_t barrier_cv;
} _pthread_vals;

static inline void* _thread_run(void* arg)
{
    const int thread_id = (unsigned long int) arg;
    const int network_indx_start = thread_id * (NUM_CELLS / NUM_THREADS);
    const int network_indx_end = network_indx_start + (NUM_CELLS / NUM_THREADS) - 1;
    
    while(_pthread_vals.curr_step < SIMUL_LEN)
	{
		for (int i = network_indx_start; i < network_indx_end; i++)
		{
			simul_fxn(_pthread_vals.network[i],
                      _pthread_vals.network,
                      _pthread_vals.curr_time,
                      _pthread_vals.curr_step);
		}

        pthread_mutex_lock(&_pthread_vals.barrier_mutx);
        _pthread_vals.num_done++;
        if (_pthread_vals.num_done < NUM_THREADS)
        {
            pthread_cond_wait(&_pthread_vals.barrier_cv,
                              &_pthread_vals.barrier_mutx);
        } else {
            _pthread_vals.curr_step++;
            _pthread_vals.curr_time += DT;
            _pthread_vals.num_done = 0;
            pthread_cond_broadcast(&_pthread_vals.barrier_cv);
        }
        pthread_mutex_unlock(&_pthread_vals.barrier_mutx);
	}
    return NULL;
}
#endif /* NUM_THREADS > 1 */

// CUDA Kernel
#ifdef CUDA
__global__ void myriad_cuda_simul(void* network[NUM_CELLS])
{
    int i = threadIdx.x;
}
#endif

int main(void)
{
    srand(42);

#ifdef MYRIAD_ALLOCATOR
    int num_allocs = 0;
    const size_t total_mem_usage = calc_total_size(&num_allocs);
    if (myriad_alloc_init(total_mem_usage, num_allocs))
    {
        fputs("Unable to initialize myriad allocator\n", stderr);
        exit(EXIT_FAILURE);
    }
    DEBUG_PRINTF("total size: %lu, num allocs: %i\n", total_mem_usage, num_allocs);
    if (atexit((void (*)(void)) myriad_finalize))
    {
        fputs("Could not set myriad_finalize to run at exit\n", stderr);
        myriad_finalize();
        exit(EXIT_FAILURE);
    }
#endif /* MYRIAD_ALLOCATOR */

	initMechanism();
    initCompartment();
	initDCCurrMech();
	initHHLeakMechanism();
	initHHNaCurrMechanism();
	initHHKCurrMechanism();
	initHHSpikeGABAAMechanism();
	initHHSomaCompartment();

	void* network[NUM_CELLS] = {NULL};
    
    const uint_fast32_t num_connxs = NUM_CELLS;
    int_fast32_t to_connect[num_connxs];

	for (int_fast32_t my_id = 0; my_id < NUM_CELLS; my_id++)
	{
        memset(to_connect, 0, sizeof(int_fast32_t) * num_connxs);
        
		// All-to-All
        for (int_fast32_t j = 0; j < NUM_CELLS; j++)
        {
            if (j == my_id)
            {
                to_connect[j] = -1;  // Don't connect to ourselves
            } else {
                to_connect[j] = j;   // Connect to cell j
            }
            DEBUG_PRINTF("to_connect[%" PRIiFAST64 "]: %" PRIiFAST64 "\n", j, to_connect[j]);
        }
        
        const bool stimulate = rand() % 2 == 0;
	    network[my_id] = new_dsac_soma(my_id,
                                       to_connect,
                                       stimulate,
                                       num_connxs);
	}

#if NUM_THREADS > 1
    // Pthread parallelism
    pthread_t _threads[NUM_THREADS];

    // Initialize global pthread values
    _pthread_vals.network = network;
    _pthread_vals.curr_time = DT;
    _pthread_vals.curr_step = 1;
    _pthread_vals.num_done = 0;
    pthread_mutex_init(&_pthread_vals.barrier_mutx, NULL);
    pthread_cond_init(&_pthread_vals.barrier_cv, NULL);

    for(unsigned long int i = 0; i < NUM_THREADS; ++i)
    {
        if(pthread_create(&_threads[i], NULL, &_thread_run, (void*) i))
        {
            DEBUG_PRINTF(stderr, "Could not create thread %lu\n", i);
            exit(EXIT_FAILURE);
        }
    }
    DEBUG_PRINTF("Done creating %d threads\n", NUM_THREADS);
    for(int i = 0; i < NUM_THREADS; ++i)
    {
        if(pthread_join(_threads[i], NULL))
        {
            DEBUG_PRINTF(stderr, "Could not join thread %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
#else
    double current_time = DT;
    for (uint_fast32_t curr_step = 1; curr_step < SIMUL_LEN; curr_step++)
    {
        for (uint_fast32_t i = 0; i < NUM_CELLS; i++)
        {
            simul_fxn(network[i], network, current_time, curr_step);
        }
        current_time += DT;
    }
#endif /* NUM_THREADS > 1 */

#ifdef MYRIAD_ALLOCATOR
    myriad_finalize();
#endif
    
    exit(EXIT_SUCCESS);
}

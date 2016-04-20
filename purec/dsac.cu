#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>
#include <tgmath.h>
#include <dirent.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#if NUM_THREADS > 1
#include <omp.h>
#endif

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
#include "HHGradedGABAAMechanism.h"
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

static inline void* new_dsac_soma(
    unsigned int id,
    int_fast32_t* connect_to,
    bool stimulate,
    const unsigned int num_connxs)
{
	void* hh_comp_obj = NULL;
    if (!(hh_comp_obj = myriad_new(HHSomaCompartment, id, 0, NULL, NULL, INIT_VM, CM)))
    {
        fputs("Failed to create mechanism", stderr);
        exit(EXIT_FAILURE);
    }

    void* hh_leak_mech = NULL;
    if (!(hh_leak_mech = myriad_new(HHLeakMechanism, id, G_LEAK, E_LEAK)))
    {
        fputs("Failed to create mechanism", stderr);
        exit(EXIT_FAILURE);
    }
    
	void* hh_na_curr_mech = NULL;
    if (!(hh_na_curr_mech = myriad_new(HHNaCurrMechanism, id, G_NA, E_NA, HH_M, HH_H)))
    {
        fputs("Failed to create mechanism", stderr);
        exit(EXIT_FAILURE);        
    }
    
	void* hh_k_curr_mech = NULL;
    if (!(hh_k_curr_mech = myriad_new(HHKCurrMechanism, id, G_K, E_K, HH_N)))
    {
        fputs("Failed to create mechanism", stderr);
        exit(EXIT_FAILURE);        
    }
    
	void* dc_curr_mech = NULL;
    if (!(dc_curr_mech = myriad_new(DCCurrentMech, id, STIM_ONSET, STIM_END, stimulate ? STIM_CURR : (double) 0.0)))
    {
        fputs("Failed to create mechanism", stderr);
        exit(EXIT_FAILURE);                
    }

	const int result =
        add_mechanism(hh_comp_obj, hh_leak_mech)
        || add_mechanism(hh_comp_obj, hh_na_curr_mech)
        || add_mechanism(hh_comp_obj, hh_k_curr_mech)
        || add_mechanism(hh_comp_obj, dc_curr_mech)
        ;
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
            
        void* GABA_mech = myriad_new(
            HHGradedGABAAMechanism,
            connect_to[i],
            0.0,
            GABA_G_MAX,
            GABA_THETA,
            GABA_SIGMA,
            GABA_TAU_ALPHA,
            GABA_TAU_BETA,
            GABA_REV);
        if (!GABA_mech || add_mechanism(hh_comp_obj, GABA_mech))
        {
            fputs("Unable to add GABA current mechanism\n", stderr);
            exit(EXIT_FAILURE);
        }
        DEBUG_PRINTF("GABA synapse from ID# %li -> #ID %u\n", connect_to[i], id);
    }

    ((struct HHSomaCompartment*)hh_comp_obj)->vm[0] = INIT_VM;

	return hh_comp_obj;
}

// CUDA Kernel
#ifdef CUDA
__global__ void myriad_cuda_simul(void* network[NUM_CELLS])
{
    int i = threadIdx.x;
    i++;
}
#endif

int main(void)
{
#ifdef MYRIAD_ALLOCATOR
    const size_t total_size = 0 +
        // Classss
        sizeof(struct MyriadObject) + sizeof(struct MyriadClass) +
        sizeof(struct Mechanism) + sizeof(struct MechanismClass) +
        sizeof(struct DCCurrentMech) + sizeof(struct DCCurrentMechClass) +
        sizeof(struct HHLeakMechanism) + sizeof(struct HHLeakMechanismClass) +
        sizeof(struct HHNaCurrMechanism) + sizeof(struct HHNaCurrMechanismClass) +
        sizeof(struct HHKCurrMechanism) + sizeof(struct HHKCurrMechanismClass) +
        sizeof(struct HHSpikeGABAAMechanism) + sizeof(struct HHSpikeGABAAMechanismClass) +
        sizeof(struct Compartment) + sizeof(struct CompartmentClass) +
        sizeof(struct HHSomaCompartment) + sizeof(struct HHSomaCompartmentClass) +
        // Objects
        (sizeof(struct HHSomaCompartment) * NUM_CELLS) +
        (sizeof(struct DCCurrentMech) * NUM_CELLS) +
        (sizeof(struct HHLeakMechanism) * NUM_CELLS) +
        (sizeof(struct HHNaCurrMechanism) * NUM_CELLS) +
        (sizeof(struct HHKCurrMechanism) * NUM_CELLS) +
        (sizeof(struct HHSpikeGABAAMechanism) * NUM_CELLS * NUM_CELLS); 
    const int num_allocs = (9 * 2) + (6 * NUM_CELLS) + (NUM_CELLS * NUM_CELLS);

    if (myriad_alloc_init(total_size, num_allocs))
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
    
    srand(42);
    
	initMechanism();
    initCompartment();
	initDCCurrMech();
	initHHLeakMechanism();
	initHHNaCurrMechanism();
	initHHKCurrMechanism();
	initHHGradedGABAAMechanism();
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
    DEBUG_PRINTF("Network status: %s\n", network[0] ? "ready": "not ready");

    double current_time = DT;
    for (uint_fast32_t curr_step = 1; curr_step < SIMUL_LEN; curr_step++)
    {
#if NUM_THREADS > 1
        #pragma omp parallel for
#endif        
        for (uint_fast32_t i = 0; i < NUM_CELLS; i++)
        {
            simul_fxn(network[i], network, current_time, curr_step);
        }
        current_time += DT;
    }

    DEBUG_PRINTF("Simulation completed at time %li\n", time(NULL));

    DEBUG_PRINTF("Writing %u files ..\n", NUM_CELLS);
    
    // Make directory if it doesn't exist
    int mkdir_result = mkdir("cmyriad_dat/", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (mkdir_result == -1 && errno != EEXIST)
    {
        perror("Couldn't create directory cmyriad_dat/: ");
        exit(EXIT_FAILURE);
    }
    
    // Change directory into data directory
    if (-1 == chdir("cmyriad_dat/"))
    {
        perror("Couldn't change directory into cmyriad_dat: ");
        exit(EXIT_FAILURE);
    }
    
#ifdef SAVE_OUTPUT
    // Write each compartment to a file
    for (uint_fast32_t comp_id = 0; comp_id < NUM_CELLS; comp_id++)
    {
        // Parametrize file name
        const size_t filename_len = sizeof("cmyriad_000.dat");
        char filename[filename_len];
        snprintf(filename, filename_len, "cmyriad_%03" PRIuFAST32 ".dat", comp_id);

        FILE* file = NULL;
        if (!(file = fopen(filename, "w+")))
        {
            perror("Couldn't open cmyriad.dat: ");
            exit(EXIT_FAILURE);
        }
#ifdef ASYNC_IO
        struct sigevent aio_sigeven;
        aio_sigeven.sigev_notify = SIGEV_NONE;

        struct aiocb async_io_params;
        async_io_params.aio_fildes = fileno(file);
        async_io_params.aio_offset = 0;
        async_io_params.aio_reqprio = 0;
        async_io_params.aio_buf = ((struct HHSomaCompartment*) network[comp_id])->vm;
        async_io_params.aio_nbytes = sizeof(double) * SIMUL_LEN;
        async_io_params.aio_sigevent = aio_sigeven;

        if (aio_write(&async_io_params) == -1)
        {
            perror("AIO Write failed immediately: ");
            exit(EXIT_FAILURE);
        }
#else
        if (1 != fwrite(((struct HHSomaCompartment*) network[comp_id])->vm,
                        sizeof(double) * SIMUL_LEN,
                        1,
                        file))
        {
            perror("Synchronous fwrite failed: ");
            exit(EXIT_FAILURE);
        }

        if (fclose(file))
        {
            perror("Could not close file: ");
            exit(EXIT_FAILURE);
        }
#endif  /* ASYNC_IO */
    }
#endif // SAVE_OUTPUT
    DEBUG_PRINTF("Writing to %" PRIuFAST32 " files scheduled/done.\n", NUM_CELLS);

    exit(EXIT_SUCCESS);
}

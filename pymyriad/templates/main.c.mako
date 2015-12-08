#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <unistd.h>
#include <pthread.h>

#ifdef CUDA
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

// Myriad C API Headers
#ifdef __cplusplus
extern "C" {
#endif

#include "myriad.h"

% for header in main_local_includes:
#include "${header}.h"
% endfor

#include "mmq.h"
    
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
#ifdef USE_DDTABLE
double _exp(double y)
{
    _eco.n.i = EXP_A * (y) + (1072693248 - EXP_C);
    return _eco.d;
}
#endif
#endif

#ifndef MYRIAD_ALLOCATOR
static ssize_t calc_total_size(int* num_allocs) __attribute__((unused));
#endif
static ssize_t calc_total_size(int* num_allocs)
{
    ssize_t total_size = 0;
    
    ## Class memory overhead calculation
% for myriad_class in simul_classes:
    total_size += sizeof(struct ${myriad_class.obj_name}) +
                  sizeof(struct ${myriad_class.cls_name});
% endfor

    *num_allocs = *num_allocs + (${len(simul_classes)} * 2);

    ## TODO: Calculate mechanism and compartment contributions to memory overhead
    total_size += 0;
    *num_allocs = *num_allocs + 0;

    // DDTABLE
#ifdef USE_DDTABLE
    *num_allocs = *num_allocs + 1;
    total_size += sizeof(struct ddtable);
    total_size += sizeof(int_fast8_t) * DDTABLE_NUM_KEYS;
    total_size += 2 * sizeof(double) * DDTABLE_NUM_KEYS;
#endif
    
    return total_size;
}

#ifdef USE_DDTABLE
ddtable_t exp_table = NULL;
#endif /* USE_DDTABLE */

#if NUM_THREADS > 1
struct _pthread_vals
{
    void** network;
    double curr_time;
    int curr_step;
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
        #pragma GCC ivdep
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


///////////////////
// Main function //
///////////////////

int main(void)
{
#ifdef MYRIAD_ALLOCATOR
    int num_allocs = 0;
    const size_t total_mem_usage = calc_total_size(&num_allocs);
    assert(myriad_alloc_init(total_mem_usage, num_allocs) == 0);
#endif /* MYRIAD_ALLOCATOR */

#ifdef USE_DDTABLE
    exp_table = ddtable_new(DDTABLE_NUM_KEYS);
#endif /* USE_DDTABLE */

#ifdef CUDA
    const bool use_cuda = true;
#else
    const bool use_cuda = false;
#endif

    ## Call init functions
% for myriad_class in simul_classes:
    init${myriad_class.obj_name}();
% endfor

	void* network[NUM_CELLS];
    
    const unsigned int num_connxs = NUM_CELLS;
    int64_t to_connect[num_connxs];

    ## TODO: Instantiate new cells with myriad_new(), add compartments, etc.

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
            fprintf(stderr, "Could not create thread %lu\n", i);
            return EXIT_FAILURE;
        }
    }
    for(int i = 0; i < NUM_THREADS; ++i)
    {
        if(pthread_join(_threads[i], NULL))
        {
            fprintf(stderr, "Could not join thread %d\n", i);
            return EXIT_FAILURE;
        }
    }
#else
    double current_time = DT;
    for (int curr_step = 1; curr_step < SIMUL_LEN; curr_step++)
    {
        #pragma GCC ivdep
        for (int i = 0; i < NUM_CELLS; i++)
        {
            simul_fxn(network[i], network, current_time, curr_step);
        }
        current_time += DT;
    }
#endif /* NUM_THREADS > 1 */

    // Cleanup
#ifdef USE_DDTABLE
    ddtable_free(exp_table);
#endif

    // Do IPC with parent python process
    struct mmq_connector conn =
        {
            .msg_queue = mmq_init_mq(),
            .socket_fd = mmq_socket_init(true, NULL),
            .connection_fd = -1,
            .server = true
        };

    // Main message loop
    while (1)
    {
        // Reset message buffer
        char* msg_buff = (char*) calloc(MMQ_MSG_SIZE + 1, sizeof(char));

        ///////////////////////////////
        // PHASE 1: SEND OBJECT SIZE //
        ///////////////////////////////

        // Wait for first message
        puts("Waiting for object request message on queue...");
        ssize_t msg_size = mq_receive(conn.msg_queue,
                                      msg_buff,
                                      MMQ_MSG_SIZE,
                                      NULL);
        if (msg_size < 0)
        {
            perror("mq_receive:");
            exit(EXIT_FAILURE);
        }
        
        // Process message for object request
        int64_t obj_req = 0;
        memcpy(&obj_req, msg_buff, MMQ_MSG_SIZE);
        printf("Object data request: %" PRIi64 "\n", obj_req);
        if (obj_req == -1)
        {
            puts("Terminating simulation.");
            break;
        }

        // Send size of compartment object & wait for it to be accepted
        size_t obj_size = myriad_size_of(network[obj_req]);
        memset(msg_buff, 0, MMQ_MSG_SIZE + 1);
        memcpy(msg_buff, &obj_size, sizeof(size_t));
        if (mq_send(conn.msg_queue, msg_buff, MMQ_MSG_SIZE, 0) != 0)
        {
            perror("mq_send size");
            exit(EXIT_FAILURE);
        }
        printf("Sent data on object size (size is %lu)\n", obj_size);

        ///////////////////////////////
        // PHASE 2: SEND OBJECT DATA //
        ///////////////////////////////
        
        // Send object data
        mmq_send_data(&conn, (unsigned char*) network[obj_req], obj_size);
        puts("Sent object data.");

        /////////////////////////////////////////////
        // PHASE 3: SEND MECHANISM DATA ONE-BY-ONE //
        /////////////////////////////////////////////
        
        const struct Compartment* as_cmp = (const struct Compartment*) network[obj_req];
        printf("Sending information for %i mechanisms.\n", as_cmp->num_mechs);
        const int my_num_mechs = as_cmp->num_mechs;
        for (int i = 0; i < my_num_mechs; i++)
        {
            // Send mechanism size data
            size_t mech_size = myriad_size_of(as_cmp->my_mechs[i]);
            if (mmq_send_data(&conn, &mech_size, sizeof(size_t)) != sizeof(mech_size))
            {
                fprintf(stderr, "Could not send mechanism %i size \n", i);
            } else {
                printf("Sent mechanism %i size of %lu.\n", i, mech_size);
            }

            // Send mechanism object
            if (mmq_send_data(&conn, as_cmp->my_mechs[i], mech_size) != (ssize_t) mech_size)
            {
                fprintf(stderr, "Could not send mechanism %i\n", i);
            } else {
                printf("Sent mechanism %i completely.\n", i);
            }
        }

        puts("Sent all mechanism objects");

        free(msg_buff);
    }
    
    puts("Exited message loop.");
    
#ifdef MYRIAD_ALLOCATOR
    assert(myriad_finalize() == 0);
#endif
    
    return 0;
}

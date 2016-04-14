<% from inspect import getmro %>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <inttypes.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>

% if CUDA:
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
% endif

## Common included header
#include "myriad.h"

% for header in dependencies:
#include "${header.__name__}.h"
% endfor

## Myriad communicator header for communicating with parent process
#include "myriad_communicator.h"
    
% if CUDA:
    % for header in dependencies:
#include "${header.__name__}.cuh"
    % endfor
% endif

## Myriad new definition
void* myriad_new(const void* _class, ...)
{
    const struct MyriadObjectClass* prototype_class = (const struct MyriadObjectClass*) _class;
    struct MyriadObject* curr_obj;
    va_list ap;

    assert(prototype_class && prototype_class->size);
    
    curr_obj = (struct MyriadObject*) _my_calloc(1, prototype_class->size);
    assert(curr_obj);

    curr_obj->mclass = (struct MyriadObjectClass*)prototype_class;

    va_start(ap, _class);
    curr_obj = (struct MyriadObject*) ctor(curr_obj, &ap);
    va_end(ap);
    
    return curr_obj;
}

## Fast exponential function structure/function
% if FAST_EXP:
__thread union _eco _eco;
% endif

static ssize_t calc_total_size(int* num_allocs)
{
    ssize_t total_size = 0;
    
    ## Class memory overhead calculation
% for myriad_class in dependencies:
    total_size += sizeof(struct ${myriad_class.obj_name}) +
                  sizeof(struct ${myriad_class.cls_name});
% endfor
    *num_allocs = *num_allocs + (${len(dependencies)} * 2);

    ## Calculate mechanism and compartment contributions to memory overhead
% for comp in compartments:
    % for cls in getmro(comp.__class__)[0:2]:
    total_size += sizeof(struct ${cls.__name__});
    % endfor
% endfor
    *num_allocs = *num_allocs + ${len(mechanisms) + len(compartments)};

    return total_size;
}

% if NUM_THREADS > 1:
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
% endif


//////////////////////////////
// Cleanup Global Variables //
//////////////////////////////

static int serversock_fd = -1, socket_fd = -1;

static void cleanup_conn(void)
{
    if (serversock_fd > 0)
    {
        m_close_socket(serversock_fd);
        serversock_fd = -1;
        unlink(UNSOCK_NAME);
    }
    if (socket_fd > 0)
    {
        m_close_socket(socket_fd);
        socket_fd = -1;
    }
}

## Handle SIGINT and SIGTERM by exiting cleanly, if possible
static void handle_signal(int signo)
{
    exit(EXIT_FAILURE);
}


///////////////////
// Main function //
///////////////////

int main(void)
{
## Setup signal handler
	if (signal(SIGTERM, handle_signal) == SIG_ERR ||
        signal(SIGINT, handle_signal) == SIG_ERR)
    {
		perror("signal failed: ");
		exit(EXIT_FAILURE);
	}

## TODO: Do srand() with provided seed, or use time()

## Setup atexit cleanup functions
    if (atexit(&cleanup_conn))
    {
        fputs("Cannot set cleanup_conn to run at exit.\n", stderr);
        exit(EXIT_FAILURE);
    }
    if (atexit((void (*)(void)) &myriad_finalize))
    {
        fputs("Cannot set myriad_finalize to run at exit.\n", stderr);
        exit(EXIT_FAILURE);
    }

## Initialize server socket so that we can accept connections
    if ((serversock_fd = m_server_socket_init(1)) == -1)
    {
        fputs("Unable to initialize server socket. Exiting.\n", stderr);
        exit(EXIT_FAILURE);
    }

    int num_allocs = 0;
    const size_t total_mem_usage = calc_total_size(&num_allocs);
    assert(myriad_alloc_init(total_mem_usage, num_allocs) == 0);

    ## Call init functions
% for myriad_class in dependencies:
    init${myriad_class.obj_name}();
% endfor

  	void* network[NUM_CELLS];

    ## TODO: Instantiate new cells with myriad_new(), add mechanisms, etc.
    int_fast32_t c_count = 0;
% for comp in compartments:
    network[c_count] = myriad_new(${comp.__class__.__name__}
    % for param in getattr(comp, "myriad_new_params").keys():
                 ,${str(getattr(comp, param))}
    % endfor
    );
    c_count++;
% endfor

% if NUM_THREADS > 1:
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
% else:
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
% endif

## Do IPC with parent python process
    if ((socket_fd = m_server_socket_accept(serversock_fd)) == -1)
    {
        fputs("Unable to accept incoming connection\n", stderr);
        exit(EXIT_FAILURE);
    }

## Main message loop
    while (1)
    {
        ///////////////////////////////
        // PHASE 1: SEND OBJECT SIZE //
        ///////////////////////////////

        // Process message for object request
        int obj_req = -1;
        if (m_receive_int(socket_fd, &obj_req) || obj_req < 0)
        {
            fputs("Terminating simulation.\n", stderr);
            exit(EXIT_FAILURE);
        }
        printf("Object data request: %d\n", obj_req);

        // Send size of compartment object & wait for it to be accepted
        const size_t obj_size = myriad_size_of(network[obj_req]);
        if (m_send_int(socket_fd, obj_size))
        {
            fputs("Failed to send object size via socket.\n", stderr);
            exit(EXIT_FAILURE);
        }
        printf("Sent data on object size (size is %lu)\n", obj_size);

        ///////////////////////////////
        // PHASE 2: SEND OBJECT DATA //
        ///////////////////////////////
        
        // Send object data
        if (m_send_data(socket_fd, network[obj_req], obj_size) < 0)
        {
            fputs("Serialization aborted: m_send_data failed\n", stderr);
            exit(EXIT_FAILURE);
        }
        puts("Sent object data.");

        /////////////////////////////////////////////
        // PHASE 3: SEND MECHANISM DATA ONE-BY-ONE //
        /////////////////////////////////////////////
        
        const struct Compartment* as_cmp = (const struct Compartment*) network[obj_req];
        printf("Sending information for %" PRIu64 " mechanisms.\n", as_cmp->num_mechs);
        const uint64_t my_num_mechs = as_cmp->num_mechs;
        for (uint64_t i = 0; i < my_num_mechs; i++)
        {
            printf("Sending information for mechanism %" PRIu64 "\n", i);
            
            // Send mechanism size
            size_t mech_size = myriad_size_of(as_cmp->my_mechs[i]);
            if (m_send_int(socket_fd, mech_size))
            {
                fputs("Failed to send Mechanism size via socket.\n", stderr);
                exit(EXIT_FAILURE);
            }
            printf("Sent mechanism %" PRIu64 "'s size of %lu.\n", i, mech_size);

            // Send mechanism object data
            if (m_send_data(socket_fd, as_cmp->my_mechs[i], mech_size) != (ssize_t) mech_size)
            {
                fprintf(stderr, "Could not send mechanism %" PRIu64"\n", i);
                exit(EXIT_FAILURE);
            }
            printf("Sent mechanism %" PRIu64 " completely.\n", i);                
        }

        puts("Sent all mechanism objects; object serialization completed.");
    }
    
    puts("Exited message loop.");
    
    exit(EXIT_SUCCESS);
}

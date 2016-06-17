<% from inspect import getmro %>

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <inttypes.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <dirent.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>

% if NUM_THREADS > 1:
#include <omp.h>
% endif

% if CUDA:
#include <vector_types.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
% endif

## Fast exponential function structure/function for non-CUDA
#ifdef FAST_EXP
#ifndef CUDA
__thread union _eco _eco;
#endif
#endif

## Common included header
#include "myriad.h"

% for header in dependencies:
#include "${header.__name__}.cuh"
% endfor

## Myriad communicator header for communicating with parent process
#include "myriad_communicator.h"

## CUDA Network & Staging arrays
% if CUDA:
__constant__ struct Compartment* dnetwork[NUM_CELLS];
struct Compartment* snetwork[NUM_CELLS];
% endif

## Host-side network array
struct Compartment* hnetwork[NUM_CELLS];

## Size-of vtable and function
const size_t size_vtable[NUM_CU_CLASS] = {
% for myriad_class in myriad_classes:
    sizeof(struct ${myriad_class.obj_name}),
% endfor
};

size_t myriad_sizeof(void* obj)
{
    return size_vtable[((struct MyriadObject*) obj)->class_id];
}

## Myriad new definition
void* myriad_new(const enum MyriadClass mclass, ...)
{
    va_list ap;
    ## Allocate object
    struct MyriadObject* new_obj = (struct MyriadObject*) calloc(1, size_vtable[mclass]);
    assert(new_obj);
    ## Assing class id
    memcpy((void*) &new_obj->class_id, &mclass, sizeof(void*));

    ## Call constructor
    va_start(ap, mclass);
    new_obj = (struct MyriadObject*) myriad_ctor(new_obj, &ap);
    assert(new_obj);
    va_end(ap);
    
    return new_obj;
}

void* myriad_cuda_new(const void* hobj)
{
% if CUDA:
    struct MyriadObject* new_obj = NULL;
    ## Allocate device object
    CUDA_CHECK_CALL(cudaMalloc(&new_obj, myriad_sizeof(hobj)));
    assert(new_obj);

    ## CUDAfy then return device pointer
    myriad_cudafy(hobj, new_obj);
    return new_obj;
% else:
    fputs("CUDA object creation is not supported when CUDA is not enabled.\\n\", stderr);
    return NULL;
% endif
}

void myriad_cuda_delete(void* hobj, void* dobj)
{
% if CUDA:
    ## Call decudafy
    myriad_decudafy(hobj, dobj);
    ## Finally, free the device object
    CUDA_CHECK_CALL(cudaFree(dobj));
% else:
    fputs("CUDA object deletion is not supported when CUDA is not enabled.\\n\", stderr);
% endif
}

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

##############################
## Cleanup Global Variables ##
##############################

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

##########################
## Simulation functions ##
##########################

% if CUDA:
__global__ void run_simul(uint_fast32_t cstep, double gtime)
{
    __shared__ unsigned int scstep;
    __shared__ double sgtime;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (i == 0)
    {
        scstep = cstep;
        sgtime = gtime;
    }
    __threadfence_block();
    while (scstep < SIMUL_LEN)
    {
        # TODO: Replace with generic compartment simul call
        HHSomaCompartment_simul<<<1, MAX_NUM_MECHS, 0, s>>>(dnetwork[i], sgtime, scstep);
        __syncthreads();
        if (i == 0)
        {
            scstep++;
            sgtime += DT;
            cudaDeviceSynchronize();
        }
        __syncthreads();
    }
}
% else:
static void run_simul(void)
{
    register double gtime = DT;
    for (uint_fast32_t cstep = 1; cstep < SIMUL_LEN; cstep++)
    {
% if NUM_THREADS > 1:
        #pragma omp parallel for
% endif
        for (size_t i = 0; i < NUM_CELLS; i++)
        {
            compartment_simul(hnetwork[i], gtime, cstep);
        }
        gtime += DT;
    }
}
% endif

################################################################
## Initialize network and copy to CUDA device, if appropriate ##
################################################################

static inline void init_network(void)
{
    ## Initialize CUDA vtables
% for myriad_class in dependencies:
    % for method in myriad_class.own_methods:
        init_${method.ident}_cuvtable();
    % endfor
% endfor

    ## Allocate and initialize host objects, copying them to the device
    size_t id = 0;
% for comp in compartments:    
    ## TODO: Initialize mechanisms 'hosted' by this compartment
    void* mechs[MAX_NUM_MECHS] = {NULL};
    size_t j = 0;
    mechs[j++] = myriad_new(HHLEAKMECHANISM, id, G_LEAK, E_LEAK);

    ## Initialize compartment with mechanisms
    ## hnetwork[id] = myriad_new(HHSOMACOMPARTMENT, j, (void**) mechs, INIT_VM, CM);
    hnetwork[id] = myriad_new(${comp.__class__.__name__.upper()}, j
    % for param in getattr(comp, "myriad_new_params").keys():
                 ,${str(getattr(comp, param))}
    % endfor
    );
    id++;
% endfor
    
    ## Copy staging network array to device network array
% if CUDA:
    for (size_t id = 0; id < NUM_CELLS; id++)
    {
        snetwork[id] = myriad_cuda_new((struct MyriadObject*) hnetwork[id]);
    }
    CUDA_CHECK_CALL(cudaMemcpyToSymbol(dnetwork, snetwork, NUM_CELLS * sizeof(void*)));
% endif
}

###################
## Main function ##
###################

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
        fputs("Cannot set cleanup_conn to run at exit.\\n", stderr);
        exit(EXIT_FAILURE);
    }
    if (atexit((void (*)(void)) &myriad_finalize))
    {
        fputs("Cannot set myriad_finalize to run at exit.\\n", stderr);
        exit(EXIT_FAILURE);
    }

    ## Initialize server socket so that we can accept connections
    if ((serversock_fd = m_server_socket_init(1)) == -1)
    {
        fputs("Unable to initialize server socket. Exiting.\\n", stderr);
        exit(EXIT_FAILURE);
    }

    ## Initialize allocator
    int num_allocs = 0;
    const size_t total_mem_usage = calc_total_size(&num_allocs);
    assert(myriad_alloc_init(total_mem_usage, num_allocs) == 0);

    ## Instantiate new cells with myriad_new(), add mechanisms, etc.
    init_network();

    ## Invoke simulation kernel
% if CUDA:
    const dim3 block(NUM_CELLS);
	const dim3 grid(NUM_CELLS / block.x);
    ## fprintf(stderr, "Execution configuration <<<%d, %d>>>\\n", grid.x, block.x);
    run_simul<<<grid, block>>>(1, DT);
    CUDA_CHECK_CALL(cudaDeviceSynchronize());
    ## Copy objects back to host-side & free staging array
    for (size_t i = 0; i < NUM_CELLS; i++)
    {
        myriad_cuda_delete(
            (struct MyriadObject*) hnetwork[i],
            (struct MyriadObject*) snetwork[i]);
    }
% else:
    run_simul();
% endif

    ## Do IPC with parent python process
    if ((socket_fd = m_server_socket_accept(serversock_fd)) == -1)
    {
        fputs("Unable to accept incoming connection\\n", stderr);
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
            fputs("Terminating simulation.\\n", stderr);
            exit(EXIT_FAILURE);
        }
        printf("Object data request: %d\\n", obj_req);

        // Send size of compartment object & wait for it to be accepted
        const size_t obj_size = myriad_sizeof(network[obj_req]);
        if (m_send_int(socket_fd, obj_size))
        {
            fputs("Failed to send object size via socket.\\n", stderr);
            exit(EXIT_FAILURE);
        }
        printf("Sent data on object size (size is %lu)\\n", obj_size);

        ///////////////////////////////
        // PHASE 2: SEND OBJECT DATA //
        ///////////////////////////////
        
        // Send object data
        if (m_send_data(socket_fd, network[obj_req], obj_size) < 0)
        {
            fputs("Serialization aborted: m_send_data failed\\n", stderr);
            exit(EXIT_FAILURE);
        }
        puts("Sent object data.");

        /////////////////////////////////////////////
        // PHASE 3: SEND MECHANISM DATA ONE-BY-ONE //
        /////////////////////////////////////////////
        
        const struct Compartment* as_cmp = (const struct Compartment*) network[obj_req];
        printf("Sending information for %" PRIu64 " mechanisms.\\n", as_cmp->num_mechs);
        const uint64_t my_num_mechs = as_cmp->num_mechs;
        for (uint64_t i = 0; i < my_num_mechs; i++)
        {
            printf("Sending information for mechanism %" PRIu64 "\\n", i);
            
            // Send mechanism size
            size_t mech_size = myriad_sizeof(as_cmp->my_mechs[i]);
            if (m_send_int(socket_fd, mech_size))
            {
                fputs("Failed to send Mechanism size via socket.\\n", stderr);
                exit(EXIT_FAILURE);
            }
            printf("Sent mechanism %" PRIu64 "'s size of %lu.\\n", i, mech_size);

            // Send mechanism object data
            if (m_send_data(socket_fd, as_cmp->my_mechs[i], mech_size) != (ssize_t) mech_size)
            {
                fprintf(stderr, "Could not send mechanism %" PRIu64"\\n", i);
                exit(EXIT_FAILURE);
            }
            printf("Sent mechanism %" PRIu64 " completely.\\n", i);                
        }

        puts("Sent all mechanism objects; object serialization completed.");
    }
    
    puts("Exited message loop.");
    
    exit(EXIT_SUCCESS);
}

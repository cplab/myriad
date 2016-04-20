/**
 * @file   myriad.h
 * @author Pedro Rittner <pr273@cornell.edu>
 * @date   Feb 17 2015
 * @brief  Master header for Myriad simulation parameters and options.
 *
 *
 * Copyright (c) 2015 Pedro Ritter <pr273@cornell.edu>
 * 
 * This file is free software: you may copy, redistribute and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 * 
 * This file is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */
#ifndef MYRIAD_H
#define MYRIAD_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 1
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

// Simulation parameters
#define DT  0.025
#define SIMUL_LEN 20000
#define NUM_CELLS 100
#define MAX_NUM_MECHS 128
#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif
// DC Stimulation params
#define STIM_ONSET 2000
#define STIM_END 6000
#define STIM_CURR  9.0
// Leak params
#define G_LEAK  1.0
#define E_LEAK  -65.0
// Sodium params
#define G_NA  35.0
#define E_NA  55.0
#define HH_M  0.01
#define HH_H  0.9
// Potassium params
#define G_K  9.0
#define E_K  -90.0
#define HH_N  0.1
// Compartment Params
#define CM  1.0
#define INIT_VM  -65.0
// GABA-a Params
#define GABA_VM_THRESH  0.0
#define GABA_THETA -5.0
#define GABA_SIGMA 0.3
#define GABA_G_MAX  1.0
#define GABA_TAU_ALPHA  0.08333333333333333
#define GABA_TAU_BETA  10.0
#define GABA_REV  -75.0

//! Use myriad's own private allocator scheme
#ifdef MYRIAD_ALLOCATOR
#define _my_malloc(size) myriad_malloc(size, true)
#define _my_calloc(num, size) myriad_calloc(num, size, true)
#define _my_free(loc) myriad_free(loc)
#else
#define _my_malloc(size) malloc(size)
#define _my_calloc(num, size) calloc(num, size)
#define _my_free(loc) free(loc)
#endif

//! Fast exponential function, as per Schraudolph 1999
#ifdef FAST_EXP
union _eco
{
    double d;
    struct _anon
    {
        int j, i;
    } n;
};
extern __thread union _eco _eco;  //! Must be thread-local due to side-effects.
#define EXP_A 1512775
#define EXP_C 60801
#define EXP(y) (_eco.n.i = EXP_A*(y) + (1072693248 - EXP_C), _eco.d)
#else
// If not using fast exponential, just alias tgmath.h exponential function
#include <tgmath.h>
#define EXP exp
#endif /* FAST_EXP */

#ifdef DEBUG
	//! Prints debug information string to stderr with file and line info.
    #define DEBUG_PRINTF(str, ...) fprintf(stderr, str, __VA_ARGS__)
#else
    #define DEBUG_PRINTF(...) do {} while (0)
#endif

#ifdef CUDA
//! Checks the return value of a CUDA library call for errors, exits if error
#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        fprintf(stderr, "Error %s at line %d in file %s\n",                 \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);       \
        exit(EXIT_FAILURE);                                                 \
    } }
#endif

//! Myriad Methods
enum MyriadMethods {
    COMPARTMENT_SIMUL = 42,
    COMPARTMENT_ADD_MECH,
    MECH_SIMUL,
    CTOR,
    DTOR,
    CUDAFY,
    DECUDAFY,
};

#endif  // MYRIAD_H

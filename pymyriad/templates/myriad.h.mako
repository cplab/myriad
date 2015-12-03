/**
 * @file   myriad.h
 * @author Pedro Rittner <pr273@cornell.edu>
 * @brief  Master header for Myriad simulation parameters and options.
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

## Necessary for gnu99
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

## Use myriad's own private allocator scheme
#ifdef MYRIAD_ALLOCATOR
#include "myriad_alloc.h"
#define _my_malloc(size) myriad_malloc(size, true)
#define _my_calloc(num, size) myriad_calloc(num, size, true)
#define _my_free(loc) myriad_free(loc)

#else

#define _my_malloc(size) malloc(size)
#define _my_calloc(num, size) calloc(num, size)
#define _my_free(loc) free(loc)
#endif

## Fast exponential function, as per Schraudolph 1999
#ifdef FAST_EXP
union _eco
{
    double d;
    struct _anon
    {
        int j, i;
    } n;
};
extern __thread union _eco _eco;  ## Must be thread-local due to side-effects.
#define EXP_A 1512775
#define EXP_C 60801

## Have to define a function in case of DDTABLE, since it uses a fxn ptr.
#ifdef USE_DDTABLE
extern double _exp(const double x);
#else
#define _exp(y) (_eco.n.i = EXP_A*(y) + (1072693248 - EXP_C), _eco.d)
#endif /* USE_DDTABLE */

#else

## If not using fast exponential, just alias math.h exponential function
#define _exp_helper exp
#define _exp _exp_helper

#endif /* FAST_EXP */


## Use hash table for exponential function lookups, with default number of keys
#ifdef USE_DDTABLE

#ifndef DDTABLE_NUM_KEYS
#define DDTABLE_NUM_KEYS 67108864
#endif  /* DDTABLE_NUM_KEYS*/

#include "ddtable.h"
extern ddtable_t exp_table;
#define EXP(x) ddtable_check_get_set(exp_table, x, &_exp)

#else

#define EXP(x) _exp(x)

#endif /* USE_DDTABLE */


## Simulation parameters

#define NUM_THREADS ${NUM_THREADS}
#define SIMUL_LEN ${SIMUL_LEN}
#define DT ${DT}
#define NUM_CELLS ${NUM_CELLS}
#define MAX_NUM_MECHS ${MAX_NUM_MECHS}


## CUDA includes (note: this has only been tested up to 6.5)
#ifdef CUDA

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

//! Checks the return value of a CUDA library call for errors, exits if error
#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        fprintf(stderr, "Error %s at line %d in file %s\n",                 \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);       \
        exit(EXIT_FAILURE);                                                 \
    } }

#endif

## Extra debug macros
#ifdef DEBUG
	//! Prints debug information string to stdout with file and line info.
    #define DEBUG_PRINTF(str, ...) do {  \
        fprintf(stdout, "DEBUG @ " __FILE__ ":" __LINE__ ": "#str, __VA_ARGS__); \
	} while(0)
#else
    #define DEBUG_PRINTF(...) do {} while (0)
#endif

#endif  // MYRIAD_H

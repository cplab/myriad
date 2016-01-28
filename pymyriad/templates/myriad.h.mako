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

typedef void (* voidf) (void);

## Use myriad's own private allocator scheme
#include "myriad_alloc.h"
#define _my_malloc(size) myriad_malloc(size, true)
#define _my_calloc(num, size) myriad_calloc(num, size, true)
#define _my_free(loc) myriad_free(loc)

## Fast exponential function, as per Schraudolph 1999
% if FAST_EXP:
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

% else:

## If not using fast exponential, just alias math.h exponential function
#define _exp_helper exp
#define _exp _exp_helper

% endif

#define EXP(x) _exp(x)


## Simulation parameters

#define NUM_THREADS ${NUM_THREADS}
#define SIMUL_LEN ${SIMUL_LEN}
#define DT ${DT}
#define NUM_CELLS ${NUM_CELLS}
#define MAX_NUM_MECHS ${MAX_NUM_MECHS}


## CUDA includes (note: this has only been tested up to 6.5)
% if CUDA:

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

## Checks the return value of a CUDA library call for errors, exits if error
#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        fprintf(stderr, "Error %s at line %d in file %s\n",                 \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);       \
        exit(EXIT_FAILURE);                                                 \
    } }

% endif

## Extra debug macros
% if DEBUG:
//! Prints debug information string to stdout with file and line info.
#define DEBUG_PRINTF(str, ...) do {  \
    fprintf(stdout, "DEBUG @ " __FILE__ ":" __LINE__ ": "#str, __VA_ARGS__); \
} while(0)
% else:
#define DEBUG_PRINTF(...) do {} while (0)
% endif

#endif  // MYRIAD_H

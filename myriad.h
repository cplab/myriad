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
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>
#include <inttypes.h>

#include <sys/resource.h>

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
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

//! Global variable for tracking Myriad's memlock management.
struct myriad_memlock_stat
{
    rlim64_t memlock_pro_lim;  //! Myriad's maximum memory locked data, in bytes
    size_t curr_memlock_usg;   //! Myriad's current memory lockage, in bytes
} myriad_memlock_stat;

//! Math function caching
#define EXP(x) exp(x)

// Unit testing macros
#ifdef UNIT_TEST
    /*! 
	 * Allows for unit testing arbitrary functions as long as they follow the
     * standard format of 0 = PASS, anything else = FAIL 
	 */
    #define UNIT_TEST_FUN(fun, ...) do {                                       \
        puts("TESTING: "#fun);                                                 \
        fprintf(stdout, #fun":\t%s\n\n", fun( __VA_ARGS__) ? "FAIL" : "PASS"); \
    } while(0)
		
	//! Compares the value of two expressions, 0 = PASS, o.w. FAIL
    #define UNIT_TEST_VAL_EQ(a,b) do {                         \
		printf("TESTING: "#a" == "#b" ... ");				   \
		fprintf(stdout, "%s\n", a == b ? "PASS" : "FAIL"); \
	} while(0)
#else
    #define UNIT_TEST_FUN(...) do {} while(0)
    #define UNIT_TEST_VAL_EQ(...) do {} while(0)
#endif

#ifdef DEBUG
	//! Prints debug information string to stdout with file and line info.
    #define DEBUG_PRINTF(str, ...) do {  \
	    fprintf(stdout, "DEBUG @ "__FILE__":"__LINE__": "#str, __VA_ARGS__) \
	} while(0)
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

#endif  // MYRIAD_H

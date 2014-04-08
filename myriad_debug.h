/**
   @file    myriad_debug.h
 
   @brief   Definitions for debugging macros and functions.
 
   @details Defines useful general debugging macros for both CPU and CUDA code.
 
   @author  Pedro Rittner
 
   @date    April 7, 2014
 
 */
#ifndef MYRIAD_DEBUG_H
#define MYRIAD_DEBUG_H

#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>
#include <assert.h>

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

//! Checks the return value of a CUDA library call for errors, exits if error
#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        fprintf(stderr, "Error %s at line %d in file %s\n",                 \
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);       \
        exit(EXIT_FAILURE);                                                 \
    } }

#endif

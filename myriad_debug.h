#ifndef MYRIAD_DEBUG_H
#define MYRIAD_DEBUG_H

#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>

#ifdef UNIT_TEST
    // Allows for unit testing arbitrary functions as long as they follow the
    // standard format of 0 = PASS, anything else = FAIL
    #define UNIT_TEST_FUN(fun, ...) do {                                        \
        puts("TESTING: "#fun);                                                  \
        fprintf(stdout, #fun":\t%s\n\n", fun( __VA_ARGS__) ? "FAIL" : "PASS");  \
    } while(0)
	// Compares the value of two expressions, 0 = PASS, o.w. FAIL
    #define UNIT_TEST_VAL_EQ(a,b) do {                         \
		printf("TESTING: "#a" == "#b" ... ");				   \
		fprintf(stdout, "%s\n", a == b ? "PASS" : "FAIL"); \
	} while(0)
#else
    #define UNIT_TEST_FUN(...) do {} while(0)
    #define UNIT_TEST_VAL_EQ(...) do {} while(0)
#endif

#endif

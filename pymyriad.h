#ifndef PYMYRIAD_H
#define PYMYRIAD_H

#include <python3.4/Python.h>
#include "MyriadObject.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DEFERRED_ADDRESS
#define DEFERRED_ADDRESS(ADDR) 0
#endif

//! Total number of C API pointers
#define PyMyriad_API_pointers 1

#ifdef PYMYRIAD_MODULE
// -------------------- BEGIN ----------------------------
// This section is used when compiling pymyriad.c
    
//! Pointer to internal C API Array
extern void* PyMyriad_API[PyMyriad_API_pointers];
    
// --------------------- END -----------------------------
#else
// -------------------- BEGIN ----------------------------
// This section is used in modules that use pymyriad's API

static void **PyMyriad_API;

//! Return -1 on error, 0 on success.
//! PyCapsule_Import will set an exception if there's an error.
static __attribute__((used)) int import_pymyriad(void)
{
    PyMyriad_API = (void **)PyCapsule_Import("pymyriad._C_API", 0);
    return (PyMyriad_API != NULL) ? 0 : -1;
}
    
// --------------------- END -----------------------------
#endif  // ifdef PYMYRIAD_MODULE

// Include "sub-modules"
#include "pymyriadobject.h"

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // ifndef PYMYRIAD_H

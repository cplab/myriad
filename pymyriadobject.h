#ifndef PYMYRIADOBJECT_H
#define PYMYRIADOBJECT_H

#include <python3.4/Python.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Header file for pymyriadobject */

/* C API functions */
#define PyMyriadObject_Init_NUM 0
#define PyMyriadObject_Init_RETURN PyObject*
#define PyMyriadObject_Init_PROTO (PyObject* self, PyObject* args, PyObject* kwds)

/* Total number of C API pointers */
#define PyMyriadObject_API_pointers 1


#ifdef PYMYRIADOBJECT_MODULE
/* This section is used when compiling pymyriadobject.c */

static PyMyriadObject_Init_RETURN PyMyriadObject_Init PyMyriadObject_Init_PROTO;

#else
/* This section is used in modules that use spammodule's API */

static void **PyMyriadObject_API;

#define PyMyriadObject_Init \
 (*(PyMyriadObject_Init_RETURN (*)PyMyriadObject_Init_PROTO) PyMyriadObject_API[PyMyriadObject_Init_NUM])

/* Return -1 on error, 0 on success.
 * PyCapsule_Import will set an exception if there's an error.
 */
static int import_pymyriadobject(void)
{
    PyMyriadObject_API = (void **)PyCapsule_Import("pymyriadobject._C_API", 0);
    return (PyMyriadObject_API != NULL) ? 0 : -1;
}

#endif

#ifdef __cplusplus
}
#endif

#endif

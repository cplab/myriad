#ifndef PYMYRIADOBJECT_H
#define PYMYRIADOBJECT_H

#include <python3.4/Python.h>
#include "MyriadObject.h"

//! Pointer to type object for MyriadObject
extern PyTypeObject* PyMyriadObject_type_p;

typedef struct
{
    PyObject_HEAD
    //! Class name of this object
    PyObject* classname;
    //! Pointer to extant object
    struct MyriadObject* mobject;
} PyMyriadObject;

// C API functions
#define PyMyriadObject_Init_NUM 0
#define PyMyriadObject_Init_RETURN PyObject*
#define PyMyriadObject_Init_PROTO (struct MyriadObject* ptr, \
                                   PyObject* args, \
                                   PyObject* kwds)

#ifdef PYMYRIAD_MODULE
static PyMyriadObject_Init_RETURN PyMyriadObject_Init PyMyriadObject_Init_PROTO;
#else
#define PyMyriadObject_Init \
 (*(PyMyriadObject_Init_RETURN (*)PyMyriadObject_Init_PROTO) \
  PyMyriad_API[PyMyriadObject_Init_NUM])
#endif  // ifdef PYMYRIAD_MODULE

#endif /* PYMYRIADOBJECT_H */

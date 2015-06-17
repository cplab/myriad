#include <python3.4/Python.h>
#include <python3.4/modsupport.h>
#include <python3.4/structmember.h>
#include <numpy/arrayobject.h>

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

//! Necessary for C API exporting
#define PYMYRIAD_MODULE
#include "pymyriad.h"

static PyObject* PyMyriadObject_Init(struct MyriadObject* ptr,
                                     PyObject* args,
                                     PyObject* kwds)
{
    PyMyriadObject* new_obj = NULL;
    new_obj = PyObject_New(PyMyriadObject, PyMyriadObject_type_p);
    if (new_obj == NULL)
    {
        PyObject_Free(new_obj);
        return NULL;
    }
    
    if (PyMyriadObject_type_p->tp_init((PyObject*) new_obj, args, kwds) < 0)
    {
        PyObject_Free(new_obj);
        return NULL;
    }

    new_obj->mobject = ptr;

    Py_INCREF(new_obj); // Necessary?
    return (PyObject*) new_obj;
}

// ----------------------------------------------------------------------------

PyDoc_STRVAR(pymyriad__doc__,
             "pymyriad is the hub for C Myriad objects.");

static PyMethodDef pymyriad_functions[] = {
    {NULL, NULL, 0, NULL}           /* sentinel */
};

static struct PyModuleDef pymyriadobjectmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "pymyriad",
    .m_doc = pymyriad__doc__,
    .m_size = -1,
    .m_methods = pymyriad_functions,
    .m_reload = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL
};

PyMODINIT_FUNC PyInit_pymyriad(void)
{
    _import_array();

    /***************/
    /* Ready types */
    /***************/
    if (PyType_Ready(PyMyriadObject_type_p) < 0)
    {
        return NULL;
    }

    /*************************/
    /* Add objects to module */
    /*************************/
    PyObject* m = PyModule_Create(&pymyriadobjectmodule);
    if (m == NULL)
    {
        return NULL;
    }

    /************************/
    /* C API Initialization */
    /************************/
    static void* PyMyriad_API[PyMyriad_API_pointers];
    
    // Initialize the C API pointer array
    PyMyriad_API[PyMyriadObject_Init_NUM] = (void*) PyMyriadObject_Init;

    // Create a Capsule containing the API pointer array's address
    PyObject* c_api_object = PyCapsule_New((void*) PyMyriad_API,
                                           "pymyriad._C_API",
                                           NULL);

    if (c_api_object != NULL)
    {
        PyModule_AddObject(m, "_C_API", c_api_object);
    }

    /**********************************/
    /* Add types to module as objects */
    /**********************************/
    Py_INCREF(PyMyriadObject_type_p);
    if (PyModule_AddObject(m, "PyMyriadObject",
                           (PyObject*) PyMyriadObject_type_p) < 0)
    {
        return NULL;
    }

    // Return finalized module on success
    return m;
}

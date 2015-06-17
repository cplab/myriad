#include "pymyriadobject.h"

#include "Compartment.h"
#include "Mechanism.h"

#include <stdio.h>

static PyObject* PyMechanism_source_id(PyObject* self __attribute__((unused)),
                                       PyObject* args)
{
    PyObject* ptr = NULL;
    unsigned int mechanism_num = 0;
    
    if (PyArg_ParseTuple(args, "OI", &ptr, &mechanism_num) < 0 || ptr == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "Couldn't parse arguments");
        return NULL;
    }

    // Object given to us is the Compartment 'parent'
    struct Compartment* parent = (struct Compartment*) ((PyMyriadObject*) ptr)->mobject;

    // Check for out-of-bounds indexing
    if (parent->num_mechs < mechanism_num)
    {
        PyErr_SetString(PyExc_IndexError, "Mechanism index out of bounds.");
        return NULL;
    }
    
    struct Mechanism* _self = (struct Mechanism*) parent->my_mechs[mechanism_num];

    // Check for invalid mechanism
    if (_self == NULL)
    {
        PyErr_SetString(PyExc_IndexError, "Mechanism at index is NULL.");
        return NULL;
    }
    
    return Py_BuildValue("K", _self->source_id);
}

static PyMethodDef pymechanism_functions[] = {
    {"source_id", PyMechanism_source_id, METH_VARARGS, "Get source id of Mechanism object"},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

static PyModuleDef pymechanism_module = {
    PyModuleDef_HEAD_INIT,
    "pymechanism",
    "Mechanism accessor methods.",
    -1,
    pymechanism_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_pymechanism(void)
{
    PyObject* m = PyModule_Create(&pymechanism_module);
    if (m == NULL)
    {
        return NULL;
    }

    return m;
}


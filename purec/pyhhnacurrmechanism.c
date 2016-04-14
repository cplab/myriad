#include "pymyriadobject.h"

#include "Compartment.h"

#include "HHNaCurrMechanism.h"

#include <stdio.h>

static PyObject* PyHHNaCurrMechanism_e_na(PyObject* self __attribute__((unused)),
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
        PyErr_SetString(PyExc_IndexError, "HHNaCurrMechanism index out of bounds.");
        return NULL;
    }

    // Do some fancy conversions
    struct HHNaCurrMechanism* _self = (struct HHNaCurrMechanism*)
        ((PyMyriadObject*) parent->my_mechs[mechanism_num])->mobject;

    // Check for invalid mechanism
    if (_self == NULL)
    {
        PyErr_SetString(PyExc_IndexError, "HHNaCurrMechanism at index is NULL.");
        return NULL;
    }
    
    return Py_BuildValue("d", _self->e_na);
}

static PyObject* PyHHNaCurrMechanism_g_na(PyObject* self __attribute__((unused)),
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
        PyErr_SetString(PyExc_IndexError, "HHNaCurrMechanism index out of bounds.");
        return NULL;
    }

    // Do some fancy conversions
    struct HHNaCurrMechanism* _self = (struct HHNaCurrMechanism*)
        ((PyMyriadObject*) parent->my_mechs[mechanism_num])->mobject;

    // Check for invalid mechanism
    if (_self == NULL)
    {
        PyErr_SetString(PyExc_IndexError, "HHNaCurrMechanism at index is NULL.");
        return NULL;
    }
    
    return Py_BuildValue("d", _self->g_na);
}

static PyObject* PyHHNaCurrMechanism_hh_m(PyObject* self __attribute__((unused)),
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
        PyErr_SetString(PyExc_IndexError, "HHNaCurrMechanism index out of bounds.");
        return NULL;
    }

    // Do some fancy conversions
    struct HHNaCurrMechanism* _self = (struct HHNaCurrMechanism*)
        ((PyMyriadObject*) parent->my_mechs[mechanism_num])->mobject;

    // Check for invalid mechanism
    if (_self == NULL)
    {
        PyErr_SetString(PyExc_IndexError, "HHNaCurrMechanism at index is NULL.");
        return NULL;
    }
    
    return Py_BuildValue("d", _self->hh_m);
}

static PyObject* PyHHNaCurrMechanism_hh_h(PyObject* self __attribute__((unused)),
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
        PyErr_SetString(PyExc_IndexError, "HHNaCurrMechanism index out of bounds.");
        return NULL;
    }

    // Do some fancy conversions
    struct HHNaCurrMechanism* _self = (struct HHNaCurrMechanism*)
        ((PyMyriadObject*) parent->my_mechs[mechanism_num])->mobject;

    // Check for invalid mechanism
    if (_self == NULL)
    {
        PyErr_SetString(PyExc_IndexError, "HHNaCurrMechanism at index is NULL.");
        return NULL;
    }
    
    return Py_BuildValue("d", _self->hh_h);
}

static PyMethodDef pyhhnacurrmechanism_functions[] = {
    {"e_na", PyHHNaCurrMechanism_e_na, METH_VARARGS, "Reversal potential of Na"},
    {"g_na", PyHHNaCurrMechanism_g_na, METH_VARARGS, "Synaptic conductance of Na"},
    {"hh_m", PyHHNaCurrMechanism_hh_m, METH_VARARGS, "TODO: m thing"},
    {"hh_h", PyHHNaCurrMechanism_hh_h, METH_VARARGS, "TODO: h thing"},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

static PyModuleDef pyhhnacurrmechanism_module = {
    PyModuleDef_HEAD_INIT,
    "pyhhnacurrmechanism",
    "HHNaCurrMechanism accessor methods.",
    -1,
    pyhhnacurrmechanism_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_pyhhnacurrmechanism(void)
{
    PyObject* m = PyModule_Create(&pyhhnacurrmechanism_module);
    if (m == NULL)
    {
        return NULL;
    }

    return m;
}


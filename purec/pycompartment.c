#include "pymyriadobject.h"

#include "Compartment.h"

#include <stdio.h>

static PyObject* PyCompartment_id(PyObject* self __attribute__((unused)),
                                  PyObject* args)
{
    PyObject* ptr = NULL;

    if (PyArg_ParseTuple(args, "O", &ptr) < 0 || ptr == NULL)
    {
        fprintf(stderr, "Couldn't parse tuple argument. \n");
        return NULL;
    }
    
    struct Compartment* _self = (struct Compartment*) ((PyMyriadObject*) ptr)->mobject;

    return Py_BuildValue("K", _self->id);
}

static PyObject* PyCompartment_num_mechs(PyObject* self  __attribute__((unused)),
                                         PyObject* args)
{
    PyObject* ptr = NULL;

    if (PyArg_ParseTuple(args, "O", &ptr) < 0 || ptr == NULL)
    {
        fprintf(stderr, "Couldn't parse tuple argument. \n");
        return NULL;
    }
    
    struct Compartment* _self = (struct Compartment*) ((PyMyriadObject*) ptr)->mobject;
    
    if (_self != NULL)
    {
        PyObject* lol = Py_BuildValue("K", _self->num_mechs);
        Py_XINCREF(lol);
        return lol;
    } else {
        Py_RETURN_NONE;
    }
}

static PyMethodDef pycompartment_functions[] = {
    {"id", PyCompartment_id, METH_VARARGS, "Get id of Compartment object"},
    {"num_mechs", PyCompartment_num_mechs, METH_VARARGS, "Get num_mechs of Compartment object"},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

static PyModuleDef pycompartment_module = {
    PyModuleDef_HEAD_INIT,
    "pycompartment",
    "Compartment accessor methods.",
    -1,
    pycompartment_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_pycompartment(void)
{
    PyObject* m = PyModule_Create(&pycompartment_module);
    if (m == NULL)
    {
        return NULL;
    }

    return m;
}


#include "pymyriadobject.h"
#include <numpy/arrayobject.h>

#include "HHSomaCompartment.h"

static PyObject* PyHHSomaCompartment_cm(PyObject* self __attribute__((unused)),
                                        PyObject* args)
{
    PyObject* ptr = NULL;

    if (PyArg_ParseTuple(args, "O", &ptr) < 0 || ptr == NULL)
    {
        fprintf(stderr, "Couldn't parse tuple argument. \n");
        return NULL;
    }
    
    struct HHSomaCompartment* _self =
        (struct HHSomaCompartment*) ((PyMyriadObject*) ptr)->mobject;

    return Py_BuildValue("d", _self->cm);
}

static PyObject* PyHHSomaCompartment_vm(PyObject* self __attribute__((unused)),
                                        PyObject* args)
{
    PyObject* ptr = NULL;

    if (PyArg_ParseTuple(args, "O", &ptr) < 0 || ptr == NULL)
    {
        fprintf(stderr, "Couldn't parse tuple argument. \n");
        return NULL;
    }
    
    struct HHSomaCompartment* _self =
        (struct HHSomaCompartment*) ((PyMyriadObject*) ptr)->mobject;

    npy_intp dims[1] = {SIMUL_LEN};
    PyObject* buf_arr = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, _self->vm);
    Py_XINCREF(buf_arr);

    return buf_arr;
}

static PyMethodDef pyhhsomacompartment_functions[] = {
    {"cm", PyHHSomaCompartment_cm, METH_VARARGS, "Get membrane capacitance"},
    {"vm", PyHHSomaCompartment_vm, METH_VARARGS, "Get membrane voltage array"},
    {NULL, NULL, 0, NULL}           /* sentinel */
};

static PyModuleDef pyhhsomacompartment_module = {
    PyModuleDef_HEAD_INIT,
    "pyhhsomacompartment",
    "HH Soma Compartment accessor methods.",
    -1,
    pyhhsomacompartment_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_pyhhsomacompartment(void)
{
    _import_array();
    
    PyObject* m = PyModule_Create(&pyhhsomacompartment_module);
    if (m == NULL)
    {
        return NULL;
    }

    return m;
}



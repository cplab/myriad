#include <python3.4/Python.h>
#include <python3.4/modsupport.h>
#include <python3.4/structmember.h>
#include <numpy/arrayobject.h>

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

PyDoc_STRVAR(spamdict__doc__,
             "spamdict is an example module showing how to subtype builtin types from C.");

#define DEFERRED_ADDRESS(ADDR) 0

/* spamdict -- a dict subtype */

typedef struct {
    PyDictObject dict;
    int state;
} spamdictobject;

static PyObject* spamdict_getstate(spamdictobject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ":getstate"))
    {
        return NULL;
    }
    return PyLong_FromLong(self->state);
}

static PyObject* spamdict_setstate(spamdictobject *self, PyObject *args)
{
    int state;

    if (!PyArg_ParseTuple(args, "i:setstate", &state))
    {
        return NULL;
    }
    self->state = state;
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef spamdict_methods[] = {
    {"getstate", (PyCFunction)spamdict_getstate, METH_VARARGS, PyDoc_STR("getstate() -> state")},
    {"setstate", (PyCFunction)spamdict_setstate, METH_VARARGS, PyDoc_STR("setstate(state)")},
    {NULL}, // Sentinel
};

static int spamdict_init(spamdictobject *self, PyObject *args, PyObject *kwds)
{
    if (PyDict_Type.tp_init((PyObject *)self, args, kwds) < 0)
    {
        return -1;
    }
    self->state = 0;
    return 0;
}

static PyMemberDef spamdict_members[] = {
    {"state", T_INT, offsetof(spamdictobject, state), READONLY, PyDoc_STR("an int variable for demonstration purposes")},
    {NULL}, // Sentinel
};

static PyTypeObject spamdict_type = {
    PyVarObject_HEAD_INIT(DEFERRED_ADDRESS(&PyType_Type), 0)
    "spamdict.spamdict",
    sizeof(spamdictobject),
    0,
    0,                                          /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_reserved */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    0,                                          /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    spamdict_methods,                           /* tp_methods */
    spamdict_members,                           /* tp_members */
    0,                                          /* tp_getset */
    DEFERRED_ADDRESS(&PyDict_Type),             /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)spamdict_init,                    /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
};

static PyObject* spam_bench(PyObject *self, PyObject *args)
{
    PyObject *obj, *name, *res;
    int n = 1000;
    time_t t0, t1;

    if (!PyArg_ParseTuple(args, "OS|i", &obj, &name, &n))
    {
        return NULL;
    }
    t0 = clock();
    while (--n >= 0)
    {
        res = PyObject_GetAttr(obj, name);
        if (res == NULL)
        {
            return NULL;
        }
        Py_DECREF(res);
    }
    t1 = clock();
    return PyFloat_FromDouble((double)(t1-t0) / CLOCKS_PER_SEC);
}

static PyMethodDef spamdict_functions[] = {
    {"bench",           spam_bench,     METH_VARARGS},
    {NULL,              NULL}           /* sentinel */
};

static struct PyModuleDef spamdictmodule = {
    PyModuleDef_HEAD_INIT,
    "spamdict",
    spamdict__doc__,
    -1,
    spamdict_functions,
    NULL,
    NULL,
    NULL,
    NULL
};


PyMODINIT_FUNC PyInit_spamdict(void)
{
    _import_array();
    PyObject *m;

    /* Fill in deferred data addresses.  This must be done before
       PyType_Ready() is called.  Note that PyType_Ready() automatically
       initializes the ob.ob_type field to &PyType_Type if it's NULL,
       so it's not necessary to fill in ob_type first. */
    spamdict_type.tp_base = &PyDict_Type;
    if (PyType_Ready(&spamdict_type) < 0)
        return NULL;

    m = PyModule_Create(&spamdictmodule);
    if (m == NULL)
        return NULL;

    if (PyType_Ready(&spamdict_type) < 0)
        return NULL;


    Py_INCREF(&spamdict_type);
    if (PyModule_AddObject(m, "spamdict",
                           (PyObject *) &spamdict_type) < 0)
        return NULL;
    return m;
}

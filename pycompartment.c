#ifndef PYCOMPARTMENT_C
#define PYCOMPARTMENT_C

#include <python3.4/Python.h>
#include <python3.4/modsupport.h>
#include <python3.4/structmember.h>
#include <numpy/arrayobject.h>

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef DEFERRED_ADDRESS
#define DEFERRED_ADDRESS(ADDR) 0
#endif

#ifndef MODULE_DEF
#define MODULE_DEF
#define PYCOMPARTMENT_SELF_SET
#endif

#include "pymyriadobject.c"

typedef struct
{
    //! Parent Myriad object
    PyMyriadObject parent;
    //! This compartment's unique ID number
	uint64_t id;
    //! Number of mechanisms in this compartment
	uint64_t num_mechs;
} PyCompartment;

static int PyCompartment_init(PyCompartment *self, PyObject *args, PyObject *kwds)
{
    PyObject* str_arg = NULL;
    uint64_t id, num_mechs;

    if (!PyArg_ParseTuple(args, "OKK", &str_arg, &id, &num_mechs))
    {
        return -1;
    }

    self->id = id;
    self->num_mechs = num_mechs;

    PyObject* new_args = Py_BuildValue("(O)", str_arg);
    
    if (PyMyriadObject_type.tp_init((PyObject*) self, new_args, kwds) < 0)
    {
        return -1;
    }

    Py_XDECREF(new_args);
    
    return 0;
}

static PyMethodDef PyCompartment_methods[] = {
    {NULL}, // Sentinel
};

static PyMemberDef PyCompartment_members[] = {
    {"id", T_ULONGLONG, offsetof(PyCompartment, id), 0, "Compartment id"},
    {"num_mechs", T_ULONGLONG, offsetof(PyCompartment, num_mechs), 0, "Number of mechanisms."},
    {NULL}, // Sentinel
};

PyTypeObject PyCompartment_type = {
    PyVarObject_HEAD_INIT(DEFERRED_ADDRESS(&PyType_Type), 0)
    "pycompartment.PyCompartment",
    sizeof(PyCompartment),
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "PyCompartment objects",                    /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    PyCompartment_methods,                      /* tp_methods */
    PyCompartment_members,                      /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyCompartment_init,               /* tp_init */
    0,                                          /* tp_alloc */
    0,                                          /* tp_new */
    0,                                          // tp_free
    0,                                          // tp_is_gc
    0,                                          // tp_bases
    0,                                          // tp_mro
    0,                                          // tp_cache
    0,                                          // tp_subclasses
    0,                                          // tp_weaklist
    0,                                          // tp_del
    0,                                          // tp_version_tag
    0,                                          // tp_finalize
};

// --------------------------------------------------

#ifdef PYCOMPARTMENT_SELF_SET

PyDoc_STRVAR(pycompartment__doc__,
             "pycompartment is the Myriad Compartment type C wrapper.");

static PyMethodDef pycompartment_functions[] = {
    {NULL, NULL, 0, NULL}           /* sentinel */
};

static struct PyModuleDef pycompartmentmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "pycompartment",
    .m_doc = pycompartment__doc__,
    .m_size = -1,
    .m_methods = pycompartment_functions,
    .m_reload = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL
};

PyMODINIT_FUNC PyInit_pycompartment(void)
{
    _import_array();
    
    /***************/
    /* Ready types */
    /***************/
    if (PyType_Ready(&PyMyriadObject_type) < 0)
    {
        return NULL;
    }
    
    // Fill in the deferred data address of child objects
    PyCompartment_type.tp_base = &PyMyriadObject_type;
    if (PyType_Ready(&PyCompartment_type) < 0)
    {
        return NULL;
    }

    /*****************/
    /* Create Module */
    /*****************/
    PyObject* m = PyModule_Create(&pycompartmentmodule);
    if (m == NULL)
    {
        return NULL;
    }
    
    /**********************************/
    /* Add types to module as objects */
    /**********************************/
    Py_INCREF(&PyMyriadObject_type);
    if (PyModule_AddObject(m, "PyMyriadObject", (PyObject *) &PyMyriadObject_type) < 0)
    {
        return NULL;
    }
    Py_INCREF(&PyCompartment_type);
    if (PyModule_AddObject(m, "PyCompartment", (PyObject *) &PyCompartment_type) < 0)
    {
        return NULL;
    }

    // Return finalized module on success
    return m;
}
#endif  // if PYCOMPARTMENT_SELF_SET

#endif  // PYCOMPARTMENT_C

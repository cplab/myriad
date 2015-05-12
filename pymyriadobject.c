#include <python3.4/Python.h>
#define PYMYRIADOBJECT_MODULE
#include "pymyriadobject.h"
#include <python3.4/modsupport.h>
#include <python3.4/structmember.h>
#include <numpy/arrayobject.h>

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "MyriadObject.h"

PyDoc_STRVAR(pymyriadobject__doc__,
             "pymyriadobject is a base type for other Myriad objects.");

#define DEFERRED_ADDRESS(ADDR) 0

/******************/
/* PyMyriadObject */
/******************/

typedef struct {
    PyObject_HEAD
    PyObject* classname;
    struct MyriadObject* mobject;
} PyMyriadObject;

static int PyMyriadObject_traverse(PyMyriadObject *self, visitproc visit, void *arg)
{
    int vret;

    if (self->classname) {
        vret = visit(self->classname, arg);
        if (vret != 0)
            return vret;
    }

    return 0;
}

static int PyMyriadObject_clear(PyMyriadObject *self)
{
    PyObject *tmp;

    tmp = self->classname;
    self->classname = NULL;
    Py_XDECREF(tmp);

    return 0;
}

static void PyMyriadObject_dealloc(PyMyriadObject* self)
{
    PyMyriadObject_clear(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* PyMyriadObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyMyriadObject *self;

    self = (PyMyriadObject *)type->tp_alloc(type, 0);
    if (self != NULL)
    {
        self->classname = PyUnicode_FromString("");
        if (self->classname == NULL)
        {
            Py_DECREF(self);
            return NULL;
        }

        self->mobject = NULL;
    }

    return (PyObject *)self;
}

static int PyMyriadObject_init(PyMyriadObject *self, PyObject *args, PyObject *kwds)
{
    PyObject* classname = NULL, *tmp = NULL;

    static char *kwlist[] = {"classname", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &classname))
    {
        return -1;
    }

    if (classname)
    {
        tmp = self->classname;
        Py_INCREF(classname);
        self->classname = classname;
        Py_XDECREF(tmp);
    }
    
    self->mobject = NULL;
    return 0;
}

static PyMethodDef PyMyriadObject_methods[] = {
    {NULL}, // Sentinel
};

static PyMemberDef PyMyriadObject_members[] = {
    {"classname", T_OBJECT_EX, offsetof(PyMyriadObject, classname), 0, "Class name"},
    {NULL}, // Sentinel
};

PyTypeObject PyMyriadObject_type = {
    PyVarObject_HEAD_INIT(DEFERRED_ADDRESS(&PyType_Type), 0)
    "pymyriadobject.PyMyriadObject",
    sizeof(PyMyriadObject),
    0,
    (destructor) PyMyriadObject_dealloc,        /* tp_dealloc */
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,   /* tp_flags */
    "PyMyriadObject objects",                   /* tp_doc */
    (traverseproc) PyMyriadObject_traverse,     /* tp_traverse */
    (inquiry) PyMyriadObject_clear,             /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    PyMyriadObject_methods,                     /* tp_methods */
    PyMyriadObject_members,                     /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc)PyMyriadObject_init,              /* tp_init */
    0,                                          /* tp_alloc */
    PyMyriadObject_new,                         /* tp_new */
};

/*****************/
/* PyCompartment */
/*****************/

typedef struct {
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
};

// --------------------------------------------------

static PyObject* PyMyriadObject_Init(PyObject* self, PyObject* args, PyObject* kwds)
{
    PyMyriadObject* new_obj = NULL;
    new_obj = PyObject_New(PyMyriadObject, &PyMyriadObject_type);
    if (new_obj == NULL)
    {
        // PyObject_Free(new_obj);
        return NULL;
    }
    
    if (PyMyriadObject_init(new_obj, args, kwds) < 0)
    {
        // PyObject_Free(new_obj);
        return NULL;
    }

    Py_INCREF(new_obj); // Necessary?
    return (PyObject*) new_obj;
}

static PyMethodDef pymyriadobject_functions[] = {
    {NULL, NULL, 0, NULL}           /* sentinel */
};

static struct PyModuleDef pymyriadobjectmodule = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "pymyriadobject",
    .m_doc = pymyriadobject__doc__,
    .m_size = -1,
    .m_methods = pymyriadobject_functions,
    .m_reload = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL
};


PyMODINIT_FUNC PyInit_pymyriadobject(void)
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
    static void *PyMyriadObject_API[PyMyriadObject_API_pointers];
    PyObject *c_api_object;
    
    /* Initialize the C API pointer array */
    PyMyriadObject_API[PyMyriadObject_Init_NUM] = (void *)PyMyriadObject_Init;

    /* Create a Capsule containing the API pointer array's address */
    c_api_object = PyCapsule_New((void *)PyMyriadObject_API, "pymyriadobject._C_API", NULL);

    if (c_api_object != NULL)
    {
        PyModule_AddObject(m, "_C_API", c_api_object);
    }

    /**********************************/
    /* Add types to module as objects */
    /**********************************/
    Py_INCREF(&PyMyriadObject_type);
    if (PyModule_AddObject(m, "PyMyriadObject",
                           (PyObject *) &PyMyriadObject_type) < 0)
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

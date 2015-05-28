#ifndef PYMYRIADOBJECT_C
#define PYMYRIADOBJECT_C

#include <python3.4/Python.h>
#include <python3.4/modsupport.h>
#include <python3.4/structmember.h>
#include <numpy/arrayobject.h>

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "MyriadObject.h"

//! Necessary for C API exporting
#define PYMYRIADOBJECT_MODULE
#include "pymyriadobject.h"

#ifndef DEFERRED_ADDRESS
#define DEFERRED_ADDRESS(ADDR) 0
#endif

#ifndef MODULE_DEF
#define MODULE_DEF
#define PYMYRIADOBJECT_SELF_SET
#endif

typedef struct
{
    PyObject_HEAD
    //! Class name of this object
    PyObject* classname;
    //! Pointer to extant object
    struct MyriadObject* mobject;
} PyMyriadObject;

static int PyMyriadObject_traverse(PyMyriadObject *self,
                                   visitproc visit,
                                   void *arg)
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

static PyObject* PyMyriadObject_new(PyTypeObject *type,
                                    PyObject *args __attribute__((unused)),
                                    PyObject *kwds __attribute__((unused)))
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

static int PyMyriadObject_init(PyMyriadObject *self,
                               PyObject *args,
                               PyObject *kwds)
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
    {"classname", T_OBJECT_EX, offsetof(PyMyriadObject, classname), 0, "Name"},
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,  // tp_flags
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

/*
static PyObject* PyMyriadObject_Init(struct MyriadObject* ptr,
                                     PyObject* args,
                                     PyObject* kwds)
{
    PyMyriadObject* new_obj = NULL;
    new_obj = PyObject_New(PyMyriadObject, &PyMyriadObject_type);
    if (new_obj == NULL)
    {
        // PyObject_Free(new_obj);
        return NULL;
    }
    new_obj->mobject = ptr;
    
    if (PyMyriadObject_init(new_obj, args, kwds) < 0)
    {
        // PyObject_Free(new_obj);
        return NULL;
    }

    Py_INCREF(new_obj); // Necessary?
    return (PyObject*) new_obj;
}
*/

// --------------------------------------------------
#ifdef PYMYRIADOBJECT_SELF_SET

PyDoc_STRVAR(pymyriadobject__doc__,
             "pymyriadobject is a base type for other Myriad objects.");

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
    /*    
    static void* PyMyriadObject_API[PyMyriadObject_API_pointers];
    PyObject* c_api_object;
    
    // Initialize the C API pointer array
    PyMyriadObject_API[PyMyriadObject_Init_NUM] = (void*) PyMyriadObject_Init;

    // Create a Capsule containing the API pointer array's address
    c_api_object = PyCapsule_New((void*) PyMyriadObject_API,
                                 "pymyriadobject._C_API",
                                 NULL);

    if (c_api_object != NULL)
    {
        PyModule_AddObject(m, "_C_API", c_api_object);
    }
    */

    /**********************************/
    /* Add types to module as objects */
    /**********************************/
    Py_INCREF(&PyMyriadObject_type);
    if (PyModule_AddObject(m, "PyMyriadObject",
                           (PyObject*) &PyMyriadObject_type) < 0)
    {
        return NULL;
    }

    // Return finalized module on success
    return m;
}

#endif  // ifdef PYMYRIADOBJECT_SELF_SET

#endif  // PYMYRIADOBJECT_C

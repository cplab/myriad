#include <python3.4/Python.h>
#include <python3.4/modsupport.h>
#include <python3.4/structmember.h>

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "pymyriad.h"

static int PyMyriadObject_traverse(PyMyriadObject *self,
                                   visitproc visit,
                                   void *arg)
{
    int vret;

    if (self->classname)
    {
        vret = visit(self->classname, arg);
        if (vret != 0)
        {
            return vret;
        }
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
    PyObject* classname = NULL;

    static char *kwlist[] = {"classname", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &classname))
    {
        return -1;
    }

    if (classname)
    {
        Py_INCREF(classname);
        self->classname = classname;
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

PyTypeObject* PyMyriadObject_type_p = &PyMyriadObject_type;

/**
 * @file mmqpy.c
 *
 * @author Pedro Rittner
 *
 * @brief Glue library for interacting with Myriad C code.
 */
#include <python3.4/Python.h>
#include <python3.4/modsupport.h>
#include <numpy/arrayobject.h>

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "myriad_communicator.h"

#include "pymyriad.h"
#include "Compartment.h"

// TODO: REMOVE
#include "Mechanism.h"

static int socket_fd = -1;

//! Module-level variable for connector
static PyObject* m_init(PyObject* self __attribute__((unused)),
                        PyObject* args __attribute__((unused)))
{
    if (socket_fd > 0)
    {
        PyErr_SetString(PyExc_Exception,
                        "Myriad connector already initialized and in use.\n");
        Py_RETURN_NONE;
    }

    // Initialize socket_fd
    socket_fd = m_client_socket_init();
    
    if (socket_fd == -1)
    {
        PyErr_SetString(PyExc_IOError,
                        "Unable to initialize Myriad connector.\n");
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject* m_close(PyObject* self __attribute__((unused)),
                         PyObject* args __attribute__((unused)))
{
    if (socket_fd == -1)
    {
        PyErr_SetString(PyExc_Exception,
                        "Attempting to close uninitialized Myriad connector\n");
        return NULL;
    }

    if (m_close_socket(socket_fd) || unlink(UNSOCK_NAME))
    {
        PyErr_SetString(PyExc_IOError,
                        "Unable to close Myriad connector.\n");
        return NULL;
    }

    // Reset socket
    socket_fd = -1;

    Py_RETURN_NONE;
}

static PyObject* retrieve_obj(PyObject* self __attribute__((unused)),
                              PyObject* args)
{
    int id = -1;
    if (!PyArg_ParseTuple(args, "i", &id))
    {
        return NULL;
    } else if (id < 0) {
        PyErr_BadArgument();
        return NULL;
    }

    // Post message saying what object we want
    printf("Requesting object ID %d ...", id);
    if (m_send_int(socket_fd, id))
    {
        PyErr_SetString(PyExc_IOError, "m_send_int failed");
        return NULL;
    }

    // Wait for object size data so we can allocate
    puts("Waiting for object size data... ");
    int obj_size = 0;
    if (m_receive_int(socket_fd, &obj_size) || obj_size < 1)
    {
        PyErr_SetString(PyExc_IOError, "m_receive_int failed");
        return NULL;
    }
    printf("Object size is: %d\n", obj_size);

    // Allocate space for object
    struct Compartment* new_comp = PyMem_Malloc(obj_size);

    // Request actual object data
    puts("Waiting for object data...");
    if (m_receive_data(socket_fd, (void*) new_comp, obj_size) != obj_size)
    {
        PyErr_SetString(PyExc_IOError, "m_receive_data failed");
        return NULL;
    }
    puts("... Received object data");

    // Read mechanisms one-by-one    
    printf("Recieving information for %" PRIu64 " mechanisms.\n",
           ((struct Compartment*)new_comp)->num_mechs);

    // Get all the mechanisms
    for (int64_t i = 0; i < new_comp->num_mechs; i++)
    {
        // Clear
        new_comp->my_mechs[i] = NULL;
        
        // Read size of mechanism
        int mech_size = 0;
        if (m_receive_int(socket_fd, &mech_size))
        {
            PyErr_SetString(PyExc_IOError, "m_receive_int failed");
            return NULL;            
        }
        printf("Mechanism %" PRIu64 " has size %i\n", i, mech_size);
        
        // Allocate space for mechanism on Python's stack
        void* mech_copy = PyMem_Malloc(mech_size);
        if (mech_copy == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Failed allocating Mechanism.");
            PyMem_Free(new_comp);
            return NULL;
        }

        // Copy contents
        if (m_receive_data(socket_fd, mech_copy, mech_size) != mech_size)
        {
            PyErr_SetString(PyExc_IOError, "m_receive_data failed");
            return NULL;
        }
        printf("Copied Mechanism %" PRIu64 "\n", i);

        // Make new object from copied contents
        PyObject* mech_obj = NULL, *str = NULL;
        str = Py_BuildValue("(s)", "Mechanism");
        if (str == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed creating Mechanism string");
            PyMem_Free(mech_copy);
            PyMem_Free(new_comp);
            return NULL;
        }
        Py_INCREF(str);

        // Initialize object
        mech_obj = PyMyriadObject_Init(mech_copy, str, NULL);
        if (mech_obj == NULL)
        {
            Py_XDECREF(str);
            PyMem_Free(mech_copy);
            PyMem_Free(new_comp);
            PyErr_SetString(PyExc_RuntimeError, "Failed copying Mechanism ");
            return NULL;
        }
        Py_INCREF(mech_obj);

        new_comp->my_mechs[i] = mech_obj;
    }

    // Prepare object data for export
    PyObject* p_obj = NULL, *str = NULL;
    str = Py_BuildValue("(s)", "Compartment");
    Py_INCREF(str);
    p_obj = PyMyriadObject_Init((struct MyriadObject*) new_comp,
                                str,
                                NULL);
    if (p_obj == NULL)
    {
        Py_XDECREF(str);
        PyErr_SetString(PyExc_Exception, "failed constructing new object");
        return NULL;
    }
    
    return p_obj;
}

static PyMethodDef MyriadCommMethods[] =
{
     {"retrieve_obj", retrieve_obj, METH_VARARGS, "Retrieve data from a Myriad object."},
     {"init", m_init, METH_NOARGS, "Open the Myriad connector."},
     {"close", m_close, METH_NOARGS, "Close the Myriad connector."},
     {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(docvar, "Myriad message queue code");

struct PyModuleDef myriad_comm_module_def =
{
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "myriad_comm",
    .m_doc  = docvar,
    .m_size = -1,
    .m_methods = MyriadCommMethods,
    .m_reload = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL
};
 
PyMODINIT_FUNC PyInit_myriad_comm(void)
{
    _import_array();

    import_pymyriad();
    
    PyObject* m = PyModule_Create(&myriad_comm_module_def);

    if (m == NULL)
    {
        return NULL;
    }

    return m;
}

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

#include "mmq.h"

#include "pymyriad.h"

#include "Compartment.h"

// TODO: REMOVE
#include "Mechanism.h"

//! Module-level variable for connector
static bool _my_q_init = false;
static struct mmq_connector _my_q;

static PyObject* mmqpy_init(PyObject* self __attribute__((unused)),
                            PyObject* args __attribute__((unused)))
{
    if (_my_q_init == true)
    {
        PyErr_SetString(PyExc_Exception,
                        "Myriad connector already initialized and in use.\n");
        Py_RETURN_NONE;
    }
    
    _my_q.msg_queue = mq_open(MMQ_FNAME, O_RDWR);
    _my_q.socket_fd = mmq_socket_init(false, NULL);
    _my_q.connection_fd = -1;
    _my_q.server = false;
    
    _my_q_init = true;
    
    if (_my_q.socket_fd == -1 || _my_q.msg_queue == -1)
    {
        PyErr_SetString(PyExc_IOError,
                        "Unable to initialize Myriad connector.\n");
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject* mmqpy_close(PyObject* self __attribute__((unused)),
                             PyObject* args __attribute__((unused)))
{
    if (_my_q_init == false)
    {
        PyErr_SetString(PyExc_Exception,
                        "Attempting to close uninitialized Myriad connector\n");
        return NULL;
    }

    if (unlink(MMQ_UNSOCK_NAME) != 0 || mq_unlink(MMQ_FNAME) != 0)
    {
        PyErr_SetString(PyExc_IOError,
                        "Unable to close Myriad connector.\n");
        return NULL;
    }

    _my_q_init = false;

    Py_RETURN_NONE;
}

static PyObject* terminate_simul(PyObject* self __attribute__((unused)),
                                 PyObject* args __attribute__((unused)))
{
    // Post message saying what object we want
    puts("Putting <TERMINATE> message on queue...");
    int64_t close_msg = -1;
    printf("obj_req: %" PRIi64 "\n", close_msg);
    char* msg_buff = malloc(sizeof(MMQ_MSG_SIZE));
    memcpy(msg_buff, &close_msg, sizeof(int64_t));
    if (mq_send(_my_q.msg_queue, msg_buff, sizeof(MMQ_MSG_SIZE), 0) != 0)
    {
        PyErr_SetString(PyExc_IOError, "mq_send failed");
        return NULL;
    }
    
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
    puts("Putting message on queue...");
    int64_t obj_req = id;
    printf("obj_req: %" PRIi64 "\n", obj_req);
    char* msg_buff = malloc(MMQ_MSG_SIZE + 1);
    memcpy(msg_buff, &obj_req, sizeof(int64_t));
    if (mq_send(_my_q.msg_queue, msg_buff, MMQ_MSG_SIZE, 0) != 0)
    {
        PyErr_SetString(PyExc_IOError, "mq_send failed");
        return NULL;
    }

    // Wait for object size data so we can allocate
    puts("Waiting for object size data: ");
    memset(msg_buff, 0, MMQ_MSG_SIZE + 1);
    const ssize_t msg_size = mq_receive(_my_q.msg_queue,
                                        msg_buff,
                                        MMQ_MSG_SIZE,
                                        NULL);
    if (msg_size < 0)
    {
        PyErr_SetString(PyExc_IOError, "mq_receive failed");
        return NULL;
    }
    size_t obj_size = 0;
    memcpy(&obj_size, msg_buff, MMQ_MSG_SIZE);
    printf("Object size is: %lu\n", obj_size);

    // Allocate space for object
    struct Compartment* new_comp = PyMem_Malloc(obj_size);

    // Request actual object data
    puts("Waiting for object data...");
    mmq_request_data(&_my_q, (void*) new_comp, obj_size);
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
        size_t mech_size = 0;
        mmq_request_data(&_my_q, &mech_size, sizeof(size_t));
        printf("Mechanism %" PRIu64 " has size %lu\n", i, mech_size);
        
        // Allocate space for mechanism on Python's stack
        void* mech_copy = PyMem_Malloc(mech_size);
        if (mech_copy == NULL)
        {
            PyErr_SetString(PyExc_MemoryError, "Failed allocating Mechanism.");
            PyMem_Free(new_comp);
            return NULL;
        }

        // Copy contents
        mmq_request_data(&_my_q, mech_copy, mech_size);
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

static PyMethodDef MmqpyMethods[] =
{
     {"retrieve_obj", retrieve_obj, METH_VARARGS, "Retrieve data from a Myriad object."},
     {"init", mmqpy_init, METH_NOARGS, "Open the Myriad connector."},
     {"close", mmqpy_close, METH_NOARGS, "Close the Myriad connector."},
     {"terminate_simul", terminate_simul, METH_NOARGS, "Terminates the Myriad simulation."},
     {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(docvar, "Myriad message queue code");

struct PyModuleDef mmqpy_module_def =
{
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "mmqpy",
    .m_doc  = docvar,
    .m_size = -1,
    .m_methods = MmqpyMethods,
    .m_reload = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL
};
 
PyMODINIT_FUNC PyInit_mmqpy(void)
{
    _import_array();

    import_pymyriad();
    
    memset(&_my_q, 0, sizeof(struct mmq_connector));

    PyObject* m = PyModule_Create(&mmqpy_module_def);

    if (m == NULL)
    {
        return NULL;
    }

    return m;
}

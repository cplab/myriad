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

#include "HHSomaCompartment.h"

//! Module-level variable for connector
static bool _my_q_init = false;
static struct mmq_connector _my_q;

static PyObject* say_hello(PyObject* self, PyObject* args)
{
    const char* name; 
 
    if (!PyArg_ParseTuple(args, "s", &name))
    {
        return NULL;
    }
 
    printf("Hello %s!\n", name);
 
    Py_RETURN_NONE;
}

static PyObject* mmqpy_init(PyObject* self, PyObject* args)
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

static PyObject* mmqpy_close(PyObject* self, PyObject* args)
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

    Py_RETURN_NONE;
}

static PyObject* retrieve_obj(PyObject* self, PyObject* args)
{
    int id = -1;
    if (!PyArg_ParseTuple(args, "i", &id))
    {
        return NULL;
    } else if (id < 0) {
        return NULL;
    }

    // Post message saying what object we want
    puts("Putting message on queue...");
    uint64_t obj_req = id;
    printf("obj_req: %" PRIu64 "\n", obj_req);
    char* msg_buff = malloc(sizeof(MMQ_MSG_SIZE));
    memcpy(msg_buff, &obj_req, sizeof(uint64_t));
    if (mq_send(_my_q.msg_queue, msg_buff, sizeof(MMQ_MSG_SIZE), 0) != 0)
    {
        PyErr_SetString(PyExc_IOError, "mq_send failed");
        return NULL;
    }

    puts("Waiting for object data: ");
    // Receive data of the object we requested
    struct HHSomaCompartment soma;
    mmq_request_data(&_my_q, &soma, sizeof(struct HHSomaCompartment));
    double* arr = PyMem_Malloc(sizeof(double) * SIMUL_LEN);
    memcpy(arr, &soma.vm, sizeof(double) * SIMUL_LEN);

    // Prepare array data
    npy_intp dims[1] = {SIMUL_LEN};
    PyObject* buf_arr = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT64, arr);

    return buf_arr;
}

static PyMethodDef MmqpyMethods[] =
{
     {"say_hello", say_hello, METH_VARARGS, "Greet somebody."},
     {"retrieve_obj", retrieve_obj, METH_VARARGS, "Retrieve data from a Myriad object."},
     {"mmqpy_init", mmqpy_init, METH_NOARGS, "Open the Myriad connector."},
     {"mmqpy_close", mmqpy_close, METH_NOARGS, "Close the Myriad connector"},
     {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(docvar, "TODO: Docstring");

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
    memset(&_my_q, 0, sizeof(struct mmq_connector));
    return PyModule_Create(&mmqpy_module_def);
}

## Top-level module includes
<%!
    from context import myriad
    from myriad import myriad_types
    from pycparser.c_ast import ArrayDecl
%>

## Top-level include (assume includes numpy)
#include "pymyriadobject.h"
#include <numpy/arrayobject.h>
#include "${obj_name}.h"

% for obj_var_name, obj_var_decl in obj_struct.members.items():
% if not obj_var_name.startswith("_"):
static PyObject* Py${obj_name}_${obj_var_name}
    (PyObject* self __attribute__((unused)), PyObject* args)
{
    PyObject* ptr = NULL;
    if (PyArg_ParseTuple(args, "O", &ptr) < 0 || ptr == NULL)
    {
        fprintf(stderr, "Couldn't parse tuple argument. \n");
        return NULL;
    }

    struct ${obj_name}* _self =
        (struct ${obj_name}*) ((PyMyriadObject*) ptr)->mobject;

    ## Check if pointer, if so return numpy array of type
    ## TODO: Array type should not always be NPY_FLOAT64
    % if isinstance(obj_var_decl.type, ArrayDecl):
    npy_intp dims[1] = {SIMUL_LEN};
    PyObject* buf_arr = PyArray_SimpleNewFromData(1,
                                                  dims,
                                                  NPY_FLOAT64,
                                                  _self->${obj_var_name});
    Py_XINCREF(buf_arr);
    return buf_arr;
    % else:
    return Py_BuildValue("${myriad_types.c_decl_to_pybuildarg(obj_var_decl)}",
                         _self->${obj_var_name});
    % endif
}
% endif
% endfor

static PyMethodDef py${obj_name.lower()}_functions[] = {
% for obj_var_name in obj_struct.members.keys():
% if not obj_var_name.startswith("_"):
    {"${obj_var_name}", Py${obj_name}_${obj_var_name}, METH_VARARGS, "TODO"},
% endif
% endfor
    {NULL, NULL, 0, NULL}           /* sentinel */
};

static PyModuleDef py${obj_name.lower()}_module = {
    PyModuleDef_HEAD_INIT,
    "py${obj_name.lower()}",
    "${obj_name} accessor methods.",
    -1,
    py${obj_name.lower()}_functions,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_${obj_name.lower()}(void)
{
    _import_array();  // Necessary for numpy support

    PyObject* m = PyModule_Create(&py${obj_name.lower()}_module);
    if (m == NULL)
    {
        return NULL;
    }

    return m;
}

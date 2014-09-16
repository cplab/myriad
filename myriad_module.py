#!/usr/bin/python3
"""
TODO: Docstring
"""

import copy

from collections import OrderedDict

from myriad_utils import enforce_annotations, TypeEnforcer
from myriad_mako_wrapper import MakoFileTemplate, MakoTemplate

from myriad_types import MyriadScalar, MyriadFunction, MyriadStructType
from myriad_types import MVoid


HEADER_FILE_TEMPLATE = """

## Python imports as a module-level block
<%!
    import myriad_types
%>

## Add include guards
<% include_guard = obj_name.upper() + "_H" %>
#ifndef ${include_guard}
#define ${include_guard}

## Add library includes
% for lib in lib_includes:
#include <${lib}>
% endfor

## Add local includes
% for lib in local_includes:
#include "${lib}"
% endfor

## Declare typedefs
% for method in methods:
${method.delegator.stringify_typedef()};
% endfor

## Struct forward declarations
struct ${cls_name};
struct ${obj_name};

## Module variables
% for m_var in module_vars:
    % if 'static' not in m_var.decl.storage:
extern ${m_var.stringify_decl()};
    % endif
% endfor

// Top-level functions
% for fun in functions:
extern ${fun.stringify_decl()};
% endfor
// Methods

% for method in methods:
extern ${method.delegator.stringify_decl()};
% endfor

// Super delegators

% for method in methods:
extern ${method.super_delegator.stringify_decl()};
% endfor

// Class/Object structs

${obj_struct.stringify_decl()}
${cls_struct.stringify_decl()}

#endif
"""

C_FILE_TEMPLATE = """

## Python imports as a module-level block
<%!
    import myriad_types
%>

#include "myriad_debug.h"

#include "${obj_name}.h"

////////////////////////////////////////////
// Forward declaration for static methods //
////////////////////////////////////////////

static void* MyriadObject_ctor(void* _self, va_list* app);
static int MyriadObject_dtor(void* _self);
static void* MyriadObject_cudafy(void* self_obj, int clobber);
static void MyriadObject_decudafy(void* _self, void* cuda_self);

static void* MyriadClass_ctor(void* _self, va_list* app);
static int MyriadClass_dtor(void* _self);
static void* MyriadClass_cudafy(void* _self, int clobber);
static void MyriadClass_decudafy(void* _self, void* cuda_self);

///////////////////////////////////////////////////////
// Static initalization for new()/classof() purposes //
///////////////////////////////////////////////////////

// Static, on-stack initialization of MyriadObject and MyriadClass classes
// Necessary because of circular dependencies (see comments below)
static struct MyriadClass object[] =
{
    // MyriadObject "anonymous" class
    {
        { object + 1 },              // MyriadClass is it's class
        object,                      // Superclass is itself (MyriadObject)
        NULL,                        // No device class by default
        sizeof(struct MyriadObject), // Size is effectively of pointer
        MyriadObject_ctor,           // Non-class constructor
        MyriadObject_dtor,           // Object destructor
        MyriadObject_cudafy,         // Gets on device as an object
        MyriadObject_decudafy,       // In-place update of CPU object using GPU object
    },
    // MyriadClass class
    {
        { object + 1 },             // MyriadClass is it's class
        object,                     // Superclass is MyriadObject (a Class is an Object)
        NULL,                       // No device class by default
        sizeof(struct MyriadClass), // Size includes methods, embedded MyriadObject
        MyriadClass_ctor,           // Constructor allows for prototype classes
        MyriadClass_dtor,           // Class destructor (No-Op, undefined behavior)
        MyriadClass_cudafy,         // Cudafication to avoid static init for extensions
        MyriadClass_decudafy,       // No-Op; DeCUDAfying a class is undefined behavior
    }
};

// Pointers to static class definition for new()/super()/classof() purposes
const void* MyriadObject = object;
const void* MyriadClass = object + 1;

static void* MyriadObject_ctor(void* _self, va_list* app)
{
    return _self;
}

static int MyriadObject_dtor(void* _self)
{
        free(_self);
        return EXIT_SUCCESS;
}

static void* MyriadObject_cudafy(void* self_obj, int clobber)
{
        #ifdef CUDA
        {
                struct MyriadObject* self = (struct MyriadObject*) self_obj;
                void* n_dev_obj = NULL;
                size_t my_size = myriad_size_of(self);

                const struct MyriadClass* tmp = self->m_class;
                self->m_class = self->m_class->device_class;

                CUDA_CHECK_RETURN(cudaMalloc(&n_dev_obj, my_size));

                CUDA_CHECK_RETURN(
                        cudaMemcpy(
                                n_dev_obj,
                                self,
                                my_size,
                                cudaMemcpyHostToDevice
                                )
                        );

                self->m_class = tmp;

                return n_dev_obj;
        }
        #else
        {
                return NULL;
        }
        #endif
}

static void MyriadObject_decudafy(void* _self, void* cuda_self)
{
        // We assume (for now) that the class hasn't changed on the GPU.
        // This makes this effectively a no-op since nothing gets copied back
        return;
}

//////////////////////////////////////////////
// MyriadClass-specific static methods //
//////////////////////////////////////////////

static void* MyriadClass_ctor(void* _self, va_list* app)
{
    struct MyriadClass* self = (struct MyriadClass*) _self;
    const size_t offset = offsetof(struct MyriadClass, my_ctor);

    self->super = va_arg(*app, struct MyriadClass*);
    self->size = va_arg(*app, size_t);

    assert(self->super);

    /*
     * MASSIVE TODO:
     * 
     * Since this is generics-based we want to be able to have default behavior for classes
     * that don't want to specify their own overrides; we probably then need to change this
     * memcpy to account for ALL the methods, not just the ones we like.
     * 
     * Solution: Make it absolutely sure if we're memcpying ALL the methods.
     */
    // Memcopies MyriadObject cudafy methods onto self (in case defaults aren't set)
    memcpy((char*) self + offset,
                   (char*) self->super + offset,
                   myriad_size_of(self->super) - offset);

    va_list ap;
    va_copy(ap, *app);

    voidf selector = NULL; selector = va_arg(ap, voidf);

    while (selector)
    {
        const voidf curr_method = va_arg(ap, voidf);
        if (selector == (voidf) myriad_ctor)
        {
            *(voidf *) &self->my_ctor = curr_method;
        } else if (selector == (voidf) myriad_cudafy) {
                        *(voidf *) &self->my_cudafy = curr_method;
                } else if (selector == (voidf) myriad_dtor) {
                        *(voidf *) &self->my_dtor = curr_method;
                } else if (selector == (voidf) myriad_decudafy) {
                        *(voidf *) &self->my_decudafy = curr_method;
                }
                selector = va_arg(ap, voidf);
    }

    return self;
}

static int MyriadClass_dtor(void* self)
{
    fprintf(stderr, "Destroying a Class is undefined behavior.\n");
    return EXIT_FAILURE;
}

// IMPORTANT: This is, ironically, for external classes' use only, since our 
// own initialization for MyriadClass is static and handled by initCUDAObjects
static void* MyriadClass_cudafy(void* _self, int clobber)
{
        /*
         * Invariants/Expectations: 
         *
         * A) The class we're given (_self) is fully initialized on the CPU
         * B) _self->device_class == NULL, will receive this fxn's result
         * C) _self->super has been set with (void*) SuperClass->device_class
         *
         * The problem here is that we're currently ignoring anything the 
         * extended class passes up at us through super_, and so we're only
         * copying the c_class struct, not the rest of the class. To solve this,
         * what we need to do is to:
         *
         * 1) Memcopy the ENTIRETY of the old class onto a new heap pointer
         *     - This works because the extended class has already made any 
         *       and all of their pointers/functions CUDA-compatible.
         * 2) Alter the "top-part" of the copied-class to go to CUDA
         *     - cudaMalloc the future location of the class on the device
         *     - Set our internal object's class pointer to that location
         * 3) Copy our copied-class to the device
         * 3a) Free our copied-class
         * 4) Return the device pointer to whoever called us
         *
         * Q: How do we keep track of on-device super class?
         * A: We take it on good faith that the under class has set their super class
         *    to be the visible SuperClass->device_class.
         */
        #ifdef CUDA
        {
            struct MyriadClass* self = (struct MyriadClass*) _self;

            const struct MyriadClass* dev_class = NULL;

            const size_t class_size = myriad_size_of(self); // DO NOT USE sizeof(struct MyriadClass)!

            // Allocate space for new class on the card
            CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_class, class_size));

            // Memcpy the entirety of the old class onto a new CPU heap pointer
            const struct MyriadClass* class_cpy = (const struct MyriadClass*) calloc(1, class_size);
            memcpy((void*)class_cpy, _self, class_size);

            // Embedded object's class set to our GPU class; this is unaffected by $clobber
            memcpy((void*)&class_cpy->_.m_class, &dev_class, sizeof(void*)); 

            CUDA_CHECK_RETURN(
                    cudaMemcpy(
                            (void*)dev_class,
                            class_cpy,
                            class_size,
                            cudaMemcpyHostToDevice
                            )
                    );
            free((void*)class_cpy); // Can safely free since underclasses get nothing

            return (void*) dev_class;
        }
        #else
        {
                return NULL;
        }
        #endif
}

static void MyriadClass_decudafy(void* _self, void* cuda_self)
{
        fprintf(stderr, "De-CUDAfying a class is undefined behavior. Aborted.\n");
        return;
}

/////////////////////////////////////
// Object management and Selectors //
/////////////////////////////////////

//----------------------------
//            New
//----------------------------

void* myriad_new(const void* _class, ...)
{
    const struct MyriadClass* prototype_class = (const struct MyriadClass*) _class;
    struct MyriadObject* curr_obj;
    va_list ap;

    assert(prototype_class && prototype_class->size);
    
    curr_obj = (struct MyriadObject*) calloc(1, prototype_class->size);
    assert(curr_obj);

    curr_obj->m_class = prototype_class;

    va_start(ap, _class);
    curr_obj = (struct MyriadObject*) myriad_ctor(curr_obj, &ap);
    va_end(ap);
        
    return curr_obj;
}

//----------------------------
//         Class Of
//----------------------------

const void* myriad_class_of(const void* _self)
{
    const struct MyriadObject* self = (const struct MyriadObject*) _self;
    return self->m_class;
}

//----------------------------
//         Size Of
//----------------------------

size_t myriad_size_of(const void* _self)
{
    const struct MyriadClass* m_class = (const struct MyriadClass*) myriad_class_of(_self);

    return m_class->size;
}

//----------------------------
//         Is A
//----------------------------

int myriad_is_a(const void* _self, const struct MyriadClass* m_class)
{
    return _self && myriad_class_of(_self) == m_class;
}

//----------------------------
//          Is Of
//----------------------------

int myriad_is_of(const void* _self, const struct MyriadClass* m_class)
{
    if (_self)
    {   
        const struct MyriadClass * myClass = (const struct MyriadClass*) myriad_class_of(_self);

        if (m_class != MyriadObject)
        {
            while (myClass != m_class)
            {
                if (myClass != MyriadObject)
                {
                    myClass = (const struct MyriadClass*) myriad_super(myClass);
                } else {
                    return 0;
                }
            }
        }

        return 1;
    }

    return 0;
}

//------------------------------
//   Object Built-in Generics
//------------------------------

void* myriad_ctor(void* _self, va_list* app)
{
    const struct MyriadClass* m_class = (const struct MyriadClass*) myriad_class_of(_self);

    assert(m_class->my_ctor);
    return m_class->my_ctor(_self, app);
}

int myriad_dtor(void* _self)
{
        const struct MyriadClass* m_class = (const struct MyriadClass*) myriad_class_of(_self);
        
        assert(m_class->my_dtor);
        return m_class->my_dtor(_self);
}

void* myriad_cudafy(void* _self, int clobber)
{
    const struct MyriadClass* m_class = (const struct MyriadClass*) myriad_class_of(_self);

        assert(m_class->my_cudafy);
        return m_class->my_cudafy(_self, clobber);
}

void myriad_decudafy(void* _self, void* cuda_self)
{
        const struct MyriadClass* m_class = (const struct MyriadClass*) myriad_class_of(_self);
        
        assert(m_class->my_decudafy);
        m_class->my_decudafy(_self, cuda_self);
        return;
}

///////////////////////////////
// Super and related methods //
///////////////////////////////

const void* myriad_super(const void* _self)
{
    const struct MyriadClass* self = (const struct MyriadClass*) _self;

    assert(self && self->super);
    return self->super;
}

void* super_ctor(const void* _class, void* _self, va_list* app)
{
    const struct MyriadClass* superclass = (const struct MyriadClass*) myriad_super(_class);

    assert(_self && superclass->my_ctor);
    return superclass->my_ctor(_self, app);
}

int super_dtor(const void* _class, void* _self)
{
        const struct MyriadClass* superclass = (const struct MyriadClass*) myriad_super(_class);

        assert(_self && superclass->my_dtor);
        return superclass->my_dtor(_self);
}

void* super_cudafy(const void* _class, void* _self, int clobber)
{
        const struct MyriadClass* superclass = (const struct MyriadClass*) myriad_super(_class);
        assert(_self && superclass->my_cudafy);
        return superclass->my_cudafy(_self, clobber);
}

void super_decudafy(const void* _class, void* _self, void* cuda_self)
{
        const struct MyriadClass* superclass = (const struct MyriadClass*) myriad_super(_class);
        assert(_self && superclass->my_decudafy);
        superclass->my_decudafy(_self, cuda_self);
        return;
}

///////////////////////////////////
//   CUDA Object Initialization  //
///////////////////////////////////

int initCUDAObjects()
{
        // Can't initialize if there be no CUDA
        #ifdef CUDA
        {
                ////////////////////////////////////////////////
                // Pre-allocate GPU classes for self-reference /
                ////////////////////////////////////////////////

                const struct MyriadClass *obj_addr = NULL, *class_addr = NULL;
        
                //TODO: Not sure if we need these; surely we can just use object[x].size instead?
                const size_t obj_size = sizeof(struct MyriadObject);
                const size_t class_size = sizeof(struct MyriadClass);

                // Allocate class and object structs on the GPU.
                CUDA_CHECK_RETURN(cudaMalloc((void**)&obj_addr, class_size)); 
                CUDA_CHECK_RETURN(cudaMalloc((void**)&class_addr, class_size));

                ///////////////////////////////////////////////////
                // Static initialization using "Anonymous"  Class /
                ///////////////////////////////////////////////////

                const struct MyriadClass anon_class_class = {
                        {class_addr}, // MyriadClass' class is itself
                        obj_addr,     // Superclass is MyriadObject (a Class is an Object)
                        class_addr,   // Device class is itself (since we're on the GPU)
                        class_size,   // Size is the class size (methods and all)
                        NULL,         // No constructor on the GPU
                        NULL,         // No destructor on the GPU
                        NULL,         // No cudafication; we're already on the GPU!
                        NULL,         // No decudafication; we *stay* on the GPU.
                };

                CUDA_CHECK_RETURN(
                        cudaMemcpy(
                                (void**) class_addr,
                                &anon_class_class,
                                sizeof(struct MyriadClass),
                                cudaMemcpyHostToDevice
                                )
                        );      

                // Remember to update static CPU class object
                object[1].device_class = class_addr; //TODO: Replace with memcpy?

                /////////////////////////////////////////////////////////
                // Static initialization using "Anonymous" Object Class /
                /////////////////////////////////////////////////////////
        
                const struct MyriadClass anon_obj_class = {
                        {class_addr}, // It's class is MyriadClass (on GPU, of course)
                        obj_addr,     // Superclass is itself
                        class_addr,   // Device class is it's class (since we're on the GPU)
                        obj_size,     // Size is effectively a pointer
                        NULL,         // No constructor on the GPU
                        NULL,         // No destructor on the GPU
                        NULL,         // No cudafication; we're already on the GPU!
                        NULL,         // No decudafication; we *stay* on the GPU
                };
        
                CUDA_CHECK_RETURN(
                        cudaMemcpy(
                                (void**) obj_addr,
                                &anon_obj_class,
                                sizeof(struct MyriadClass),
                                cudaMemcpyHostToDevice
                                )
                        );
        
                // Remember to update static CPU object
                object[0].device_class = (const struct MyriadClass*) obj_addr; //TODO: Replace with memcpy?

                /////////////////////////////////////////////////
                // Memcpy GPU class pointers to *_dev_t symbols /
                /////////////////////////////////////////////////

                CUDA_CHECK_RETURN(
                        cudaMemcpyToSymbol(
                                (const void*) &MyriadClass_dev_t,
                                &class_addr,
                                sizeof(void*),
                                0,
                                cudaMemcpyHostToDevice
                                )
                        );

                CUDA_CHECK_RETURN(
                        cudaMemcpyToSymbol(
                                (const void*) &MyriadObject_dev_t,
                                &obj_addr,
                                sizeof(void*),
                                0,
                                cudaMemcpyHostToDevice
                                )
                        );

                return 0;
        } 
    #else
        {
                return EXIT_FAILURE;
        }
        #endif
}

"""


# pylint: disable=R0902
# pylint: disable=R0903
class MyriadMethod(object):
    """
    Generic class for abstracting methods into 3 core components:

    1) Delegator - API entry point for public method calls
    2) Super Delegator - API entry point for subclasses
    3) Instance Function(s) - 'Internal' method definitions for each class
    """

    DELG_TEMPLATE = """
<%
    fun_args = ','.join([arg.ident for arg in delegator.args_list.values()])
%>

${delegator.stringify_decl()}
{
    const struct MyriadClass* m_class = (const struct MyriadClass*) myriad_class_of(${delegator.args_list[0].ident});

    assert(m_class->${delegator.fun_typedef.name});

    % if delegator.ret_var.base_type is MVoid and not delegator.ret_var.base_type.ptr:
    m_class->my_${delegator.fun_typedef.name}(${fun_args});
    return;
    % else:
    return m_class->my_${delegator.fun_typedef.name}(${fun_args});
    % endif
}
    """

    SUPER_DELG_TEMPLATE = """
<%
    fun_args = ','.join([arg.ident for arg in super_delegator.args_list.values()])
%>
${supert_delegator.stringify_decl()}
{
    const struct MyriadClass* superclass = (const struct MyriadClass*) myriad_super(${super_delegator.args_list[0].ident});

    assert(superclass->${delegator.fun_typedef.name});

    % if delegator.ret_var.base_type is MVoid and not delegator.ret_var.base_type.ptr:
    superclass->my_${delegator.fun_typedef.name}(${fun_args});
    return;
    % else:
    return superclass->my_${delegator.fun_typedef.name}(${fun_args});
    % endif
}
    """

    @enforce_annotations
    def __init__(self,
                 m_fxn: MyriadFunction,
                 instance_methods: dict=None):
        """
        Initializes a method from a function.

        The point of this class is to automatically create delegators for a
        method. This makes inheritance of methods easier since the delegators
        are not implemented by the subclass, only the instance methods are
        overwritten.
        """

        # Need to ensure this function has a typedef
        m_fxn.gen_typedef()
        self.delegator = m_fxn

        # Initialize (default: None) instance method
        self.instance_methods = {}
        # If we are given a string, assume this is the instance method body
        # and auto-generate the MyriadFunction wrapper.
        for obj_name, i_method in instance_methods.items():
            if type(i_method) is str:
                self.gen_instance_method_from_str(i_method, obj_name)
            else:
                raise NotImplementedError("Non-string instance methods.")

        # Create super delegator
        super_args = copy.copy(m_fxn.args_list)
        super_class_arg = MyriadScalar("_class",
                                       MVoid,
                                       True,
                                       ["const"])
        tmp_arg_indx = len(super_args)+1
        super_args[tmp_arg_indx] = super_class_arg
        super_args.move_to_end(tmp_arg_indx, last=False)
        _delg = MyriadFunction("super_" + m_fxn.ident,
                               super_args,
                               m_fxn.ret_var)
        self.super_delegator = _delg
        self.delg_template = MakoTemplate(self.DELG_TEMPLATE, vars(self))
        self.super_delg_template = MakoTemplate(self.SUPER_DELG_TEMPLATE,
                                                vars(self))
        # TODO: Implement instance method template
        self.instance_method_template = None

    def gen_instance_method_from_str(self,
                                     method_body: str,
                                     obj_name: str):
        """
        Automatically generate a MyriadFunction wrapper for a method body.
        """
        _tmp_f = MyriadFunction(obj_name + '_' + self.delegator.ident,
                                args_list=self.delegator.args_list,
                                ret_var=self.delegator.ret_var,
                                storage=['static'],
                                fun_def=method_body)
        self.instance_methods[obj_name] = _tmp_f


# pylint: disable=R0902
class MyriadModule(object, metaclass=TypeEnforcer):
    """
    Represents an independent Myriad module (e.g. MyriadObject).
    """

    DEFAULT_LIB_INCLUDES = {"stdlib.h", "stdio.h", "assert.h",
                            "stddef.h", "stdarg.h", "stdint.h"}

    DEFAULT_CUDA_INCLUDES = {"cuda_runtime.h", "cuda_runtime_api.h"}

    @enforce_annotations
    def __init__(self,
                 supermodule,
                 obj_name: str,
                 cls_name: str=None,
                 obj_vars: OrderedDict=None,
                 methods: set=None,
                 cuda: bool=False):
        """Initializes a module"""

        # Set CUDA support status
        self.cuda = cuda

        # Set internal names for classes
        self.obj_name = obj_name
        if cls_name is None:
            self.cls_name = obj_name + "Class"
        else:
            self.cls_name = cls_name

        # methods = delegator, super delegator, instance
        # TODO: Implement method setting
        self.methods = set()

        # Initialize class object and object class

        # Add implicit superclass to start of struct definition
        if obj_vars is not None:
            _arg_indx = len(obj_vars)+1
            obj_vars[_arg_indx] = supermodule.cls_struct("_", quals=["const"])
            obj_vars.move_to_end(_arg_indx, last=False)
        else:
            obj_vars = OrderedDict()
        self.obj_struct = MyriadStructType(self.obj_name, obj_vars)

        # Initialize class variables, i.e. function pointers for methods
        cls_vars = OrderedDict()
        cls_vars[0] = supermodule.cls_struct("_", quals=["const"])

        for indx, method in enumerate(self.methods):
            m_scal = MyriadScalar("my_" + method.delegator.fun_typedef.name,
                                  method.delegator.base_type)
            cls_vars[indx+1] = m_scal

        self.cls_vars = cls_vars
        self.cls_struct = MyriadStructType(self.cls_name, self.cls_vars)

        # TODO: Dictionaries or sets?
        self.functions = set()

        # Initialize module global variables
        self.module_vars = set()
        v_obj = MyriadScalar(self.obj_name,
                             MVoid,
                             True,
                             quals=["const"])
        self.module_vars.add(v_obj)
        v_cls = MyriadScalar(self.cls_name,
                             MVoid,
                             True,
                             quals=["const"])
        self.module_vars.add(v_cls)

        # Initialize standard library imports, by default with fail-safes
        self.lib_includes = MyriadModule.DEFAULT_LIB_INCLUDES

        # TODO: Initialize local header imports
        self.local_includes = set()

        # Initialize C header template
        self.header_template = None
        self.initialize_header_template()

    def register_module_function(self,
                                 function: MyriadFunction,
                                 strict: bool=False,
                                 override: bool=False):
        """
        Registers a global function in the module.

        Note: strict and override are mutually exclusive.

        Keyword arguments:
        method -- method to be registered
        strict -- if True, raises an error if a collision occurs when joining.
        override -- if True, overrides superclass methods.
        """
        if strict is True and override is True:
            raise ValueError("Flags strict and override cannot both be True.")

        # TODO: Make "override"/"strict" modes check for existance better.
        if function in self.functions:
            if strict:
                raise ValueError("Cannot add duplicate functions.")
            elif override:
                self.functions.discard(function)
        self.functions.add(function)

    def initialize_header_template(self, context_dict: dict=None):
        """ Initializes internal Mako template for C header file. """
        if context_dict is None:
            context_dict = vars(self)
        self.header_template = MakoFileTemplate(self.obj_name+".h",
                                                HEADER_FILE_TEMPLATE,
                                                context_dict)

    def render_header_template(self, printout: bool=False):
        """ Renders the header file template. """

        # Reset buffer if necessary
        if self.header_template.buffer is not '':
            self.header_template.reset_buffer()

        self.header_template.render()

        if printout:
            print(self.header_template.buffer)
        else:
            self.header_template.render_to_file()


def main():
    pass


if __name__ == "__main__":
    main()

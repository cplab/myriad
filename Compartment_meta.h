/**
   @author Alex J Davies
 */

#ifndef COMPARTMENT_META_H
#define COMPARTMENT_META_H

#include "myriad_metaprogramming.h"

// Generics
#define COMPARTMENT_OBJECT Compartment
#define COMPARTMENT_CLASS CompartmentClass

// Attributes
#define COMPARTMENT_OBJECT_SUPERCLASS MyriadObject
#define COMPARTMENT_OBJECT_SUPERCLASS_NAME _
#define COMPARTMENT_CLASS_SUPERCLASS MyriadClass
#define COMPARTMENT_CLASS_SUPERCLASS_NAME _
#define ID id
#define NUM_MECHS num_mechs
#define MY_MECHS my_mechs

// Simulation function
#define SIMUL_FXN_NAME compartment_simul_fxn_t
#define SIMUL_FXN_ARGS \
    void* self, \
    void** network, \
    const double dt, \
    const double global_time, \
    const unsigned int curr_step
#define SIMUL_FXN_RET void
#define MY_COMPARTMENT_SIMUL_FXN m_comp_fxn

// Compartment add mech
#define ADD_MECH_FXN_NAME compartment_add_mech_t
#define ADD_MECH_FXN_ARGS \
    void* self, \
    void* mechanism 
#define ADD_MECH_FXN_RET int
#define MY_COMPARTMENT_ADD_MECH_FXN m_add_mech_fun


#endif

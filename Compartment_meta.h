/**
   @author Alex J Davies
 */

#ifndef COMPARTMENT_META_H
#define COMPARTMENT_META_H

#include "myriad_metaprogramming.h"

// Generics
#define COMPARTMENT_OBJECT_NAME Compartment
#define COMPARTMENT_CLASS_NAME CompartmentClass

// Attributes
#define COMPARTMENT_OBJECT_SUPERCLASS MyriadObject
#define COMPARTMENT_CLASS_SUPERCLASS MyriadClass
#define ID id
#define NUMBER_MECHS num_mechs
#define MY_MECHS my_mechs


// Simulation function
#define SIMUL_FXN_NAME compartment_simul_fxn_t
#define NETWORK network
#define TIMESTEP dt
#define GLOBAL_TIME global_time
#define CURR_STEP curr_step
#define SIMUL_FXN_ARGS \
    void* self, \
    void** NETWORK, \
    const double TIMESTEP, \
    const double GLOBAL_TIME, \
    const unsigned int CURR_STEP
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

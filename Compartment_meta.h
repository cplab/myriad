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
#define SIMUL_FXN_NAME_T simul_fxn_t
#define SIMUL_FXN_NAME_D simul_fxn
#define MY_COMPARTMENT_SIMUL_CLASS_FXN m_comp_fxn
#define INDIVIDUAL_SIMUL_FXN_NAME simul_fxn
#define SIMUL_FXN_ARGS void* self, \
        void** network, \
        const double dt, \
        const double global_time, \
        const unsigned int curr_step
#define SUPER_SIMUL_FXN_ARGS void* _class, \
	void* _self, \
	void** network, \
	const double dt, \
	const double global_time, \
	const unsigned int curr_step
#define SIMUL_FXN_RET void


// Add mech function
#define ADDMECH_FXN_NAME_T add_mech_t
#define ADDMECH_FXN_NAME_D add_mechanism
#define MY_COMPARTMENT_ADDMECH_CLASS_FXN m_add_mech_fun
#define INDIVIDUAL_ADDMECH_FXN_NAME add_mech
#define ADDMECH_FXN_ARGS void* self, \
        void* mechanism
#define SUPER_ADDMECH_FXN_ARGS const void* _class, \
        void* self, \
        void* mechanism
#define ADDMECH_FXN_RET int

// Dynamic initialisation properties
#define COMPARTMENT_INIT_FXN_NAME initCompartment


#endif

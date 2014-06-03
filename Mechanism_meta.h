/**
   @author Alex J Davies
 */


#ifndef MECHANISM_META_H
#define MECHANISM_META_H

#include "myriad_metaprogramming.h"

// Generics
#define MECHANISM_OBJECT Mechanism
#define MECHANISM_CLASS MechanismClass

// Attributes
#define SUPERCLASS_MECHANISM_OBJECT_NAME _
#define SUPERCLASS_MECHANISM_CLASS_NAME _
#define COMPARTMENT_PREMECH_SOURCE_ID source_id

// Mechanism function
#define MECH_FXN_NAME_T mech_fun_t
#define MECH_FXN_NAME_D mechanism_fxn
#define MY_MECHANISM_MECH_CLASS_FXN m_mech_fxn
#define INDIVIDUAL_MECH_FXN_NAME mech_fun
#define MECH_FXN_ARGS void* _self, \
	void* pre_comp, \
	void* post_comp, \
	const double dt, \
	const double global_time, \
	const unsigned int curr_step
#define SUPER_MECH_FXN_ARGS void* _class, \
	void* _self, \
	void* pre_comp, \
	void* post_comp, \
	const double dt, \
	const double global_time, \
	const unsigned int curr_step
#define MECH_FXN_RET double

// Dynamic initialisation properties
#define MECHANISM_INIT_FXN_NAME initMechanism





#endif

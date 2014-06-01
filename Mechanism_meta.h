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
#define MY_MECHANISM_MECH_CLASS_FXN m_mech_fxn

// Mechanism function
#define MECH_FXN_NAME mech_fun_t
#define MECH_FXN_NAME_C mechanism_fxn
// THE ABOVE TWO MACROS NEED UNIFYING!
#define MECH_FXN_ARGS void* _self, \
	void* pre_comp, \
	void* post_comp, \
	const double dt, \
	const double global_time, \
	const unsigned int curr_step
#define MECH_FXN_RET double

// Mechanism class function
#define MECH_CLASS_FXN_NAME mechanism_fxn
#define MECH_CLASS_FXN_ARGS void* _self, \
	void* pre_comp, \
	void* post_comp, \
	const double dt, \
	const double global_time, \
	const unsigned int curr_step
#define MECH_CLASS_FXN_RET double





#endif

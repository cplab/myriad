#ifndef HHNACURRMECHANISM_META_H
#define HHNACURRMECHANISM_META_H

#include "myriad_metaprogramming.h"

// Generics
#define HHNACURRMECHANISM_OBJECT HHNaCurrMechanism
#define HHNACURRMECHANISM_CLASS HHNaCurrMechanismClass

// Attributes
#define SUPERCLASS_HHNACURRMECHANISM_OBJECT_NAME _
#define SUPERCLASS_HHNACURRMECHANISM_CLASS_NAME _
#define HHNACURRMECHANISM_CHANNEL_CONDUCTANCE g_na
#define HHNACURRMECHANISM_REVERSAL_POTENTIAL e_na
#define HHNACURRMECHANISM_HH_M hh_m
#define HHNACURRMECHANISM_HH_H hh_h

// Mechanism function
#define HHNACURRMECHANISM_MECH_FXN_RET double
#define HHNACURRMECHANISM_MECH_FXN_ARGS void* _self, \
	void* pre_comp, \
	void* post_comp, \
	const double dt, \
	const double global_time, \
	const unsigned int curr_step
#define HHNACURRMECHANISM_MECH_FXN_NAME mech_fun




#endif

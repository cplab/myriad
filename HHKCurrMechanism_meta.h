#ifndef HHKCURRMECHANISM_META_H
#define HHKCURRMECHANISM_META_H

#include "myriad_metaprogramming.h"

// Generics
#define HHKCURRMECHANISM_OBJECT HHKCurrMechanism
#define HHKCURRMECHANISM_CLASS HHKCurrMechanismClass

// Attributes
#define SUPERCLASS_HHKCURRMECHANISM_OBJECT _
#define SUPERCLASS_HHKCURRMECHANISM_CLASS _
#define HHKCURRMECHANISM_CHANNEL_CONDUCTANCE g_k
#define HHKCURRMECHANISM_REVERE_POTENTIAL e_k
#define HHKCURRMECHANISM_HH_N hh_n

// Mechanism function
#define HHKCURRMECHANISM_MECH_FXN_RET double
#define HHKCURRMECHANISM_MECH_FXN_ARGS void* _self, \
	void* pre_comp, \
	void* post_comp, \
	const double dt, \
	const double global_time, \
	const unsigned int curr_step
#define HHKCURRMECHANISM_MECH_FXN_NAME mech_fun


#endif

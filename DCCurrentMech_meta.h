#ifndef DCCURRENTMECH_META_H
#define DCCURRENTMECH_META_H

#include "myriad_metaprogramming.h"

// Generics
#define DCCURRENTMECHANISM_OBJECT DCCurrentMech
#define DCCURRENTMECHANISM_CLASS DCCurrentMechClass

// Attributes
#define SUPERCLASS_DCCURRENTMECHANISM_OBJECT_NAME _
#define SUPERCLASS_DCCURRENTMECHANISM_CLASS_NAME _
#define DCCURRENTMECHANISM_T_START t_start
#define DCCURRENTMECHANISM_T_STOP t_stop
#define DCCURRENTMECHANISM_AMPLITUDE amplitude

// Mechanism function
#define DCCURRENTMECHANISM_MECH_FXN_RET double
#define DCCURRENTMECHANISM_MECH_FXN_ARGS void* _self, \
	void* pre_comp, \
	void* post_comp, \
	const double dt, \
	const double global_time, \
	const unsigned int curr_step
#define DCCURRENTMECHANISM_MECH_FXN_NAME mech_fun

#endif


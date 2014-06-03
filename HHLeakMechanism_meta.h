#ifndef HHLEAKMECHANISM_META_H
#define HHLEAKMECHANISM_META_H

#include "myriad_metaprogramming.h"

// Generics
#define HHLEAKMECHANISM_OBJECT HHLeakMechanism
#define HHLEAKMECHANISM_CLASS HHLeakMechanismClass

// Attributes
#define SUPERCLASS_HHLEAKMECHANISM_OBJECT_NAME _
#define SUPERCLASS_HHLEAKMECHANISM_CLASS_NAME _
#define HHLEAKMECHANISM_G_LEAK g_leak
#define HHLEAKMECHANISM_E_REV e_rev

// Mechanism function
#define HHLEAKMECHANISM_MECH_FXN_RET double
#define HHLEAKMECHANISM_MECH_FXN_ARGS void* _self, \
	void* pre_comp, \
	void* post_comp, \
	const double dt, \
	const double global_time, \
	const unsigned int curr_step
#define HHLEAKMECHANISM_MECH_FXN_NAME mech_fun

// Dynamic initialisation properties
#define HHLEAKMECHANISM_INIT_FXN_NAME initHHLeakMechanism


#endif

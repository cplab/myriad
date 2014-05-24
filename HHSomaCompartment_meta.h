/**
   @author Alex J Davies
 */

#ifndef HHSOMACOMPARTMENT_META_H
#define HHSOMACOMPARTMENT_META_H

#include "myriad_metaprogramming.h"

// Generics
#define HHSOMACOMPARTMENT_OBJECT HHSomaCompartment
#define HHSOMACOMPARTMENT_CLASS HHSomaCompartmentClass

// Attributes
#define SUPERCLASS_HHSOMACOMPARTMENT_OBJECT_NAME _
#define SUPERCLASS_HHSOMACOMPARTMENT_CLASS_NAME _
#define HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE soma_vm
#define HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE_LENGTH soma_vm_len
#define HHSOMACOMPARTMENT_CAPACITANCE cm

// Simulation function
#define HHSOMACOMPARTMENT_SIMUL_FXN_RET void
#define HHSOMACOMPARTMENT_SIMUL_FXN_ARGS void* _self, \
	void** network, \
	const double dt, \
	const double global_time, \
	const unsigned int curr_step
#define HHSOMACOMPARTMENT_SIMUL_FXN_NAME simul_fxn

#endif

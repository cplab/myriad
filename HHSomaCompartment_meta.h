/**
   @author Alex J Davies
 */

#ifndef HHSOMACOMPARTMENT_META_H
#define HHSOMACOMPARTMENT_META_H

#include "myriad_metaprogramming.h"

// Generics
#define SELF_NAME self
#define SELF_TYPE void*
#define _SELF_NAME _MYRIAD_CAT(_,SELF_NAME)
#define OBJECT_NAME HHSomaCompartment
#define CLASS_NAME HHSomaCompartmentClass

// Attributes
#define COMPARTMENT_NAME _
#define MEMBRANE_VOLTAGE soma_vm
#define MEMBRANE_VOLTAGE_LENGTH soma_vm_len
#define CAPACITANCE cm


#endif

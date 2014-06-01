/**
   @file    Mechanism.h
 
   @brief   Generic Mechanism class definition file.
 
   @details Defines the generic Mechanism class specification for Myriad
 
   @author  Pedro Rittner
 
   @date    April 7, 2014
 */
#ifndef MECHANISM_H
#define MECHANISM_H

#include "MyriadObject.h"

#include "Mechanism_meta.h"

//! Mechanism function typedef
typedef MYRIAD_FXN_TYPEDEF_GEN(MECH_FXN_NAME, MECH_FXN_ARGS, MECH_FXN_RET);

// Generic pointers for new/class-of purposes

extern const void* MECHANISM_OBJECT; // myriad_new(Mechanism, ...);
extern const void* MECHANISM_CLASS;

// -----------------------------------------

/**
   Delegator function for MechanismClass mechanism function method.

   @param[in]    _self        pointer to extant object instance
   @param[in]    pre_comp     pointer to the compartment where this mechanism is
   @param[in]    dt           timestep of the simulation
   @param[in]    global_time  current global time of the simulation
   @param[in]    curr_step    current global time step of the simulation

   @returns calculated output value of this mechanism for the given timestep
 */

extern MYRIAD_FXN_METHOD_HEADER_GEN_NO_SUFFIX(MECH_CLASS_FXN_RET, MECH_CLASS_FXN_ARGS, MECH_CLASS_FXN_NAME);

// ----------------------------------------

/**
   Mechanism object structure definition.

   Stores mechanism state.

   @see MyriadObject
 */
struct MECHANISM_OBJECT
{
	const struct MyriadObject SUPERCLASS_MECHANISM_OBJECT_NAME; //! Mechanism : MyriadObject
	unsigned int COMPARTMENT_PREMECH_SOURCE_ID;      //! Source ID of the pre-mechanism compartment
};

/**
   Mechanism class structure definition.

   Defined mechanism behavior.

   @see MyriadClass
 */
struct MECHANISM_CLASS
{
	const struct MyriadClass SUPERCLASS_MECHANISM_CLASS_NAME; //! MechanismClass : MyriadClass
	MECH_FXN_NAME MY_MECHANISM_MECH_CLASS_FXN;      //! Mechanism simulation function
};

// -------------------------------------

/**
   Initializes prototype mechanism infrastructure on the heap.

   @param[in]    init_cuda    flag for directing CUDA protoype initialization
 */
void initMechanism(const int init_cuda);

#endif

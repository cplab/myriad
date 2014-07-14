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

//! Mechanism function typedef
typedef double (* mech_fun_t) (
	void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
);

// Generic pointers for new/class-of purposes

extern const void* Mechanism; // myriad_new(Mechanism, ...);
extern const void* MechanismClass;

// -----------------------------------------

/**
   Delegator function for MechanismClass mechansim function method.

   @param[in]    _self        pointer to extant object instance
   @param[in]    pre_comp     pointer to the compartment where this mechanism is
   @param[in]    dt           timestep of the simulation
   @param[in]    global_time  current global time of the simulation
   @param[in]    curr_step    current global time step of the simulation

   @returns calculated output value of this mechanism for the given timestep
 */
extern double mechanism_fxn(
	void* _self,
	void* pre_comp,
	void* post_comp,
	const double dt,
	const double global_time,
	const unsigned int curr_step
);

// ----------------------------------------

/**
   Mechanism object structure definition.

   Stores mechanism state.

   @see MyriadObject
 */
struct Mechanism
{
	const struct MyriadObject _; //! Mechanism : MyriadObject
	unsigned int source_id;      //! Source ID of the pre-mechanism compartment
};

/**
   Mechanism class structure definition.

   Defined mechanism behavior.

   @see MyriadClass
 */
struct MechanismClass
{
	const struct MyriadClass _; //! MechanismClass : MyriadClass
	mech_fun_t m_mech_fxn;      //! Mechanism simulation function
};

// -------------------------------------

/**
   Initializes prototype mechanism infrastructure on the heap.

   @param[in]    init_cuda    flag for directing CUDA protoype initialization
 */
extern void initMechanism(const int init_cuda);

#endif

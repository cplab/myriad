/**
   @file    Compartment.h
 
   @brief   Generic Compartment class definition file.
 
   @details Defines the Compartment class specification for Myriad
 
   @author  Pedro Rittner
 
   @date    April 7, 2014
 */
#ifndef COMPARTMENT_H
#define COMPARTMENT_H

#include "MyriadObject.h"
#include "Mechanism.h"

#include "Compartment_meta.h"

//! Compartment simulate function pointer
typedef MYRIAD_FXN_TYPEDEF_GEN(SIMUL_FXN_NAME, SIMUL_FXN_ARGS, SIMUL_FXN_RET);

//! Method for adding mechanisms to a compartment
typedef MYRIAD_FXN_TYPEDEF_GEN(ADD_MECH_FXN_NAME, ADD_MECH_FXN_ARGS, ADD_MECH_FXN_RET);

// Generic pointers for new/class-of purposes

extern const void* COMPARTMENT_OBJECT;
extern const void* COMPARTMENT_CLASS;

/**
   Generic simulation function delegator.

   @param[in]    _self        pointer to extant object instance
   @param[in]    network      list of pointers to other compartments in network
   @param[in]    dt           timestep of the simulation
   @param[in]    global_time  current global time of the simulation
   @param[in]    curr_step    current global time step of the simulation
 */
extern MYRIAD_FXN_METHOD_HEADER_GEN_NO_SUFFIX(SIMUL_FXN_RET, SIMUL_FXN_ARGS, simul_fxn);

/**
   Generic mechanism adder delegator.

   @param[in]    _self      pointer to extant compartent instance
   @param[in]    mechanism  pointer to extant mechanism to add
   
   @returns EXIT_SUCCESS if addition completed, EXIT_FAILURE otherwise.
*/
extern MYRIAD_FXN_METHOD_HEADER_GEN_NO_SUFFIX(ADD_MECH_FXN_RET, ADD_MECH_FXN_ARGS, add_mechanism);

//TOOD: Array of pointers for mechanisms vs Array of structs; better for performance?
//! Generic Compartment structure definition
struct COMPARTMENT_OBJECT
{
	const struct COMPARTMENT_OBJECT_SUPERCLASS COMPARTMENT_OBJECT_SUPERCLASS_NAME; //! Compartment : MyriadObject
	unsigned int ID;             //! This compartment's unique ID number
	unsigned int NUM_MECHS;      //! Number of mechanisms in this compartment
	struct Mechanism** MY_MECHS; //! List of mechanisms in this compartment
};

//! Generic CompartmentClass structure definition
struct COMPARTMENT_CLASS
{
	const struct COMPARTMENT_CLASS_SUPERCLASS COMPARTMENT_CLASS_SUPERCLASS_NAME;            //! CompartmentClass : MyriadClass
	SIMUL_FXN_NAME MY_COMPARTMENT_SIMUL_FXN;    //! Defines compartment simulation
	ADD_MECH_FXN_NAME MY_COMPARTMENT_ADD_MECH_FXN; //! Allows for adding mechanisms to compartment
};

/**
   Initializes prototype compartment infrastructure on the heap.

   @param[in]    init_cuda    flag for directing CUDA protoype initialization
 */
void initCompartment(const int init_cuda);

#endif

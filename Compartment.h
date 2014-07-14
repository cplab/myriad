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

//! Compartment simulate function pointer
typedef void (* compartment_simul_fxn_t) (
	void* _self,
	void** NETWORK,
	const double TIMESTEP,
	const double GLOBAL_TIME,
	const unsigned int CURR_STEP
	);

//! Method for adding mechanisms to a compartment
typedef int (* compartment_add_mech_t) (
	void* _self,
	void* mechanism
	);

// Generic pointers for new/class-of purposes

extern const void* Compartment;
extern const void* CompartmentClass;

/**
   Generic simulation function delegator.

   @param[in]    _self        pointer to extant object instance
   @param[in]    network      list of pointers to other compartments in network
   @param[in]    dt           timestep of the simulation
   @param[in]    global_time  current global time of the simulation
   @param[in]    curr_step    current global time step of the simulation
 */
extern void simul_fxn(
	void* _self,
	void** network,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	);

/**
   Generic mechanism adder delegator.

   @param[in]    _self      pointer to extant compartent instance
   @param[in]    mechanism  pointer to extant mechanism to add
   
   @returns EXIT_SUCCESS if addition completed, EXIT_FAILURE otherwise.
*/
extern int add_mechanism(void* _self, void* mechanism);

//TOOD: Array of pointers for mechanisms vs Array of structs; better for performance?
//! Generic Compartment structure definition
struct Compartment
{
	const struct MyriadObject _; //! Compartment : MyriadObject
	unsigned int id;             //! This compartment's unique ID number
	unsigned int num_mechs;      //! Number of mechanisms in this compartment
	struct Mechanism** my_mechs; //! List of mechanisms in this compartment
};

//! Generic CompartmentClass structure definition
struct CompartmentClass
{
	const struct MyriadClass _;            //! CompartmentClass : MyriadClass
	compartment_simul_fxn_t m_compartment_simul_fxn;    //! Defines compartment simulation
	compartment_add_mech_t m_compartment_add_mech_fxn; //! Allows for adding mechanisms to compartment
};

/**
   Initializes prototype compartment infrastructure on the heap.

   @param[in]    init_cuda    flag for directing CUDA protoype initialization
 */
extern void initCompartment(const int init_cuda);

#endif

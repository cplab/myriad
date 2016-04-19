/**
 * @file    Compartment.h
 *
 * @brief   Generic Compartment class definition file.
 *
 * @details Defines the Compartment class specification for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    April 7, 2014
 */
#ifndef COMPARTMENT_H
#define COMPARTMENT_H

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

#include "MyriadObject.h"
#include "Mechanism.h"

//! Compartment simulate function pointer
typedef void (* compartment_simul_fxn_t) (void* _self,
                                          void** network,
                                          const double global_time,
                                          const uint_fast32_t curr_step);

//! Method for adding mechanisms to a compartment
typedef int (* compartment_add_mech_t) (void* _self, void* mechanism);

//! Generic pointer for new(Compartment) purposes
extern const void* Compartment;
//! Generic pointer for new(CompartmentClass) purposes
extern const void* CompartmentClass;

//! Generic Compartment structure definition
struct Compartment
{
    //! Compartment : MyriadObject
	const struct MyriadObject _;
    //! This compartment's unique ID number
	uint_fast32_t id;
    //! Number of mechanisms in this compartment
	uint_fast32_t num_mechs;
    //! List of mechanisms in this compartment
	void* my_mechs[MAX_NUM_MECHS];
};

//! Generic CompartmentClass structure definition
struct CompartmentClass
{
    //! CompartmentClass : MyriadClass
	const struct MyriadClass _;
    //! Defines compartment simulation
	compartment_simul_fxn_t m_compartment_simul_fxn;
    //! Allows for adding mechanisms to compartment
	compartment_add_mech_t m_compartment_add_mech_fxn;
};

/**
 * Generic simulation function delegator.
 *
 * @param[in]  _self        pointer to extant object instance
 * @param[in]  network      list of pointers to other compartments in network
 * @param[in]  global_time  current global time of the simulation
 * @param[in]  curr_step    current global time step of the simulation
 */
#define simul_fxn(self, network, g_time, c_step) \
    ((const struct CompartmentClass*) myriad_class_of(self))->m_compartment_simul_fxn(self, network, g_time, c_step)

#define super_simul_fxn(class, self, network, g_time, c_step) \
    ((const struct CompartmentClass*) myriad_super(class))->m_compartment_simul_fxn(self, network, g_time, c_step)

/**
 * Generic mechanism adder delegator.
 *
 * @param[in]  _self      pointer to extant compartent instance
 * @param[in]  mechanism  pointer to extant mechanism to add
 * 
 * @returns 0 if addition completed, -1 otherwise.
*/
#define add_mechanism(self, mechanism) \
    ((const struct CompartmentClass*) myriad_class_of(self))->m_compartment_add_mech_fxn(self, mechanism)

#define super_add_mechanism(class, self, mechanism) \
    ((const struct CompartmentClass*) myriad_super(class))->(self, mechanism)

/**
 * Initializes prototype compartment infrastructure on the heap.
 */
extern void initCompartment(void);

#endif /* COMPARTMENT_H */

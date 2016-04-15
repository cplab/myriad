/**
 * @file    Mechanism.h
 *
 * @brief   Generic Mechanism class definition file.
 *
 * @details Defines the generic Mechanism class specification for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    April 7, 2014
 */
#ifndef MECHANISM_H
#define MECHANISM_H

#include <stdint.h>
#include <stdbool.h>

#include "MyriadObject.h"

//! Mechanism function typedef
typedef double (* mech_fun_t) (void* _self,
                               void* pre_comp,
                               void* post_comp,
                               const double global_time,
                               const uint64_t curr_step);

//! Generic pointer for new(Mechanism) purposes
extern const void* Mechanism;
//! Generic pointer for new(MechanismClass) purposes
extern const void* MechanismClass;

/**
 * Delegator function for MechanismClass mechansim function method.
 *
 * @param[in]  _self        pointer to extant object instance
 * @param[in]  pre_comp     pointer to the compartment where this mechanism is
 * @param[in]  global_time  current global time of the simulation
 * @param[in]  curr_step    current global time step of the simulation
 *
 * @returns calculated output value of this mechanism for the given timestep
 */
extern double mechanism_fxn(void* _self,
                            void* pre_comp,
                            void* post_comp,
                            const double global_time,
                            const uint64_t curr_step);

/**
 * Mechanism object structure definition.
 *
 * Stores mechanism state.
 *
 * @see MyriadObject
 */
struct Mechanism
{
    //! Mechanism : MyriadObject
	const struct MyriadObject _;
    //! Source ID of the pre-mechanism compartment
	uint64_t source_id;
};

/**
 * Mechanism class structure definition.
 *
 * Defined mechanism behavior.
 *
 * @see MyriadClass
 */
struct MechanismClass
{
    //! MechanismClass : MyriadClass
	const struct MyriadClass _;
    //! Mechanism simulation function
	mech_fun_t m_mech_fxn;
};

/**
 * Initializes prototype mechanism infrastructure on the heap.
 *
 * @param[in]  init_cuda  flag for directing CUDA protoype initialization
 */
extern void initMechanism(void);

#endif /* MECHANISM_H */

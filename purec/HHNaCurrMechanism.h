/**
 * @file    HHNaCurrMechanism.h
 *
 * @brief   Hodgkin-Huxley Sodium Mechanism class definition file.
 *
 * @details Defines the Hodgkin-Huxley Sodium Mechanism class specification for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    April 9, 2014
 */
#ifndef HHNACURRMECHANISM_H
#define HHNACURRMECHANISM_H

#include "MyriadObject.h"
#include "Mechanism.h"

// Generic pointers for new/class-of purposes

extern const void* HHNaCurrMechanism;
extern const void* HHNaCurrMechanismClass;

/**
 * HHNaCurrMechanism mechanism for Hodgkin-Huxley sodium channel.
 *
 * @see Mechanism
 */
struct HHNaCurrMechanism
{
    //! HHNaCurrMechanism : Mechanism
	struct Mechanism _;
    //! Sodium channel conductance - nS
	scalar g_na;
    //! Sodium reversal potential - mV
	scalar e_na;
    //! @TODO Figure out what hh_m is actually called
	scalar hh_m;
    //! @TODO Figure out what hh_h is actually called
	scalar hh_h;
};

struct HHNaCurrMechanismClass
{
    //! HHNaCurrMechanismClass : MechanismClass
	struct MechanismClass _; 
};

void initHHNaCurrMechanism(void);

#endif

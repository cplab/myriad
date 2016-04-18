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
#ifndef HHKCURRMECHANISM_H
#define HHKCURRMECHANISM_H

#include "MyriadObject.h"
#include "Mechanism.h"

// Generic pointers for new/class-of purposes

extern const void* HHKCurrMechanism;
extern const void* HHKCurrMechanismClass;

/**
 * HHKCurrMechanism mechanism for Hodgkin-Huxley potassium channel.
 *
 * @see Mechanism
 */
struct HHKCurrMechanism
{
    //! HHKCurrMechanism : Mechanism
	struct Mechanism _;
    //! Sodium channel conductance - nS
	double g_k;
    //! Sodium reversal potential - mV
	double e_k;
    //! @TODO Figure out what hh_n is actually called
	double hh_n;
};

struct HHKCurrMechanismClass
{
    //! HHKCurrMechanismClass : MechanismClass
	struct MechanismClass _;
};

void initHHKCurrMechanism(void);

#endif

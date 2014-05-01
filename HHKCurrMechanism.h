/**
   @file    HHNaCurrMechanism.h
 
   @brief   Hodgkin-Huxley Sodium Mechanism class definition file.
 
   @details Defines the Hodgkin-Huxley Sodium Mechanism class specification for Myriad
 
   @author  Pedro Rittner
 
   @date    April 9, 2014
 */
#ifndef HHKCURRMECHANISM_H
#define HHKCURRMECHANISM_H

#include "MyriadObject.h"
#include "Mechanism.h"

// Generic pointers for new/class-of purposes

extern const void* HHKCurrMechanism;
extern const void* HHKCurrMechanismClass;

/**
   HHKCurrMechanism mechanism for Hodgkin-Huxley potassium channel.

   @see Mechanism
 */
struct HHKCurrMechanism
{
	struct Mechanism _; //! HHKCurrMechanism : Mechanism
	double g_k;		    //! Sodium channel conductance - nS
	double e_k;		    //! Sodium reversal potential - mV
	double hh_n;	    //! @TODO Figure out what hh_n is actually called
};

struct HHKCurrMechanismClass
{
	struct MechanismClass _; //! HHKCurrMechanismClass : MechanismClass
};

void initHHKCurrMechanism(int cuda_init);

#endif

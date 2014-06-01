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

#include "HHKCurrMechanism_meta.h"

// Generic pointers for new/class-of purposes

extern const void* HHKCURRMECHANISM_OBJECT;
extern const void* HHKCURRMECHANISM_CLASS;

/**
   HHKCurrMechanism mechanism for Hodgkin-Huxley potassium channel.

   @see Mechanism
 */
struct HHKCURRMECHANISM_OBJECT
{
	struct MECHANISM_OBJECT SUPERCLASS_HHKCURRMECHANISM_OBJECT; //! HHKCurrMechanism : Mechanism
	double HHKCURRMECHANISM_CHANNEL_CONDUCTANCE;		    //! Sodium channel conductance - nS
	double HHKCURRMECHANISM_REVERE_POTENTIAL;		    //! Sodium reversal potential - mV
	double HHKCURRMECHANISM_HH_N;	    //! @TODO Figure out what hh_n is actually called
};

struct HHKCURRMECHANISM_CLASS
{
	struct MECHANISM_CLASS SUPERCLASS_HHKCURRMECHANISM_CLASS; //! HHKCurrMechanismClass : MechanismClass
};

void initHHKCurrMechanism(int cuda_init);

#endif

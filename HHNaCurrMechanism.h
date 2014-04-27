/**
   @file    HHNaCurrMechanism.h
 
   @brief   Hodgkin-Huxley Sodium Mechanism class definition file.
 
   @details Defines the Hodgkin-Huxley Sodium Mechanism class specification for Myriad
 
   @author  Pedro Rittner
 
   @date    April 9, 2014
 */
#ifndef HHNACURRMECHANISM_H
#define HHNACURRMECHANISM_H

#include "MyriadObject.h"
#include "Mechanism.h"

// Generic pointers for new/class-of purposes

extern const void* HHNaCurrMechanism;
extern const void* HHNaCurrMechanismClass;

/**
   HHNaCurrMechanism mechanism for Hodgkin-Huxley sodium channel.

   @see Mechanism
 */
struct HHNaCurrMechanism
{
	struct Mechanism _; //! HHNaCurrMechanism : Mechanism
	double g_na;	    //! Sodium channel conductance - nS
	double e_na;	    //! Sodium reversal potential - mV
	double hh_m;	    //! @TODO Figure out what hh_m is actually called
	double hh_h;	    //! @TODO Figure out what hh_h is actually called
};

struct HHNaCurrMechanismClass
{
	struct MechanismClass _; //! HHNaCurrMechanismClass : MechanismClass
};

void initHHNaCurrMechanism(int cuda_init);

#endif

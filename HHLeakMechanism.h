/**
   @file    HHLeakMechanism.h
 
   @brief   Hodgkin-Huxley Leak Mechanism class definition file.
 
   @details Defines the Hodgkin-Huxley Leak Mechanism class specification for Myriad
 
   @author  Pedro Rittner
 
   @date    April 9, 2014
 */
#ifndef HHLEAKMECHANISM_H
#define HHLEAKMECHANISM_H

#include "MyriadObject.h"
#include "Mechanism.h"

// Generic pointers for new/class-of purposes

extern const void* HHLeakMechanism;
extern const void* HHLeakMechanismClass;

/**
   HHLeakMechanism mechanism for Hodgkin-Huxley leak channel.

   @see Mechanism
 */
struct HHLeakMechanism
{
	struct Mechanism _; //! HHLeakMechanism : Mechanism
	double g_leak;      //! Leak Conductance - nS
	double e_rev;       //! Leak Reversal Potential - mV
};

struct HHLeakMechanismClass
{
	struct MechanismClass _; //! HHLeakMechanismClass : MechanismClass
};

void initHHLeakMechanism(int cuda_init);

#endif

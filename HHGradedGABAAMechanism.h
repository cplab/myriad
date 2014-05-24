/**
   @file    HHGradedGABAAMechanism.h
 
   @brief   Hodgkin-Huxley GABA-a Mechanism class definition file.
 
   @details Defines the Hodgkin-Huxley GABA-a Mechanism class specification for Myriad
 
   @author  Pedro Rittner
 
   @date    April 9, 2014
 */
#ifndef HHGABAACURRMECHANISM_H
#define HHGABAACURRMECHANISM_H

#include "MyriadObject.h"
#include "Mechanism.h"

#include "HHGradedGABAAMechanism_meta.h"

// Generic pointers for new/class-of purposes

extern const void* HHGRADEDGABAAMECHANISM_OBJECT;
extern const void* HHGRADEDGABAAMECHANISM_CLASS;

/**
   HHGradedGABAAMechanism mechanism for Hodgkin-Huxley GABA-a synapse.

   @see Mechanism
 */
struct HHGRADEDGABAAMECHANISM_OBJECT
{
	struct Mechanism SUPERCLASS_HHGRADEDGABAAMECHANISM_OBJECT_NAME;     //! HHGradedGABAAMechanism : Mechanism
	double* HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING;			//! Synaptic gating variable (unitless, >=0, <=1)
	unsigned int HHGRADEDGABAAMECHANISM_SYNAPTIC_GATING_LENGTH;	//! Length of the synaptic gating variable array (0 -> MAXINT)
	double HHGRADEDGABAAMECHANISM_MAX_SYN_CONDUCTANCE;			//! Maximum synaptic conductance - nS
	double HHGRADEDGABAAMECHANISM_HALF_ACTIVATION_POTENTIAL;			//! Half-activation potential - mV
	double HHGRADEDGABAAMECHANISM_MAXIMAL_SLOPE;			//! Maximal slope of the sigmoidal synaptic function
	double HHGRADEDGABAAMECHANISM_CHANNEL_OPENING_TIME; 		//! Channel opening time constant - ms
	double HHGRADEDGABAAMECHANISM_CHANNEL_CLOSING_TIME;		//! Channel closing time constant - ms
	double HHGRADEDGABAAMECHANISM_REVERSAL_POTENTIAL;		//! Synaptic reversal potential - mV
};

struct HHGRADEDGABAAMECHANISM_CLASS
{
	struct MechanismClass SUPERCLASS_HHGRADEDGABAAMECHANISM_CLASS_NAME; //! HHGradedGABAAMechanismClass : MechanismClass
};

void initHHGradedGABAAMechanism(int cuda_init);

#endif

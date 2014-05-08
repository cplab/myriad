/**
   @file    HHGABAAMechanism.h
 
   @brief   Hodgkin-Huxley GABA-a Mechanism class definition file.
 
   @details Defines the Hodgkin-Huxley GABA-a Mechanism class specification for Myriad
 
   @author  Pedro Rittner
 
   @date    April 9, 2014
 */
#ifndef HHGABAACURRMECHANISM_H
#define HHGABAACURRMECHANISM_H

#include "MyriadObject.h"
#include "Mechanism.h"

// Generic pointers for new/class-of purposes

extern const void* HHGABAAMechanism;
extern const void* HHGABAAMechanismClass;

/**
   HHGABAAMechanism mechanism for Hodgkin-Huxley GABA-a synapse.

   @see Mechanism
 */
struct HHGABAAMechanism
{
	struct Mechanism _;     //! HHGABAAMechanism : Mechanism
	double* g_s;			//! Synaptic gating variable (unitless, >=0, <=1)
	unsigned int g_s_len;	//! Length of the synaptic gating variable array (0 -> MAXINT)
	double g_max;			//! Maximum synaptic conductance - nS
	double theta;			//! Half-activation potential - mV
	double sigma;			//! Maximal slope of the sigmoidal synaptic function
	double tau_alpha; 		//! Channel opening time constant - ms
	double tau_beta;		//! Channel closing time constant - ms
	double gaba_rev;		//! Synaptic reversal potential - mV
};

struct HHGABAAMechanismClass
{
	struct MechanismClass _; //! HHGABAAMechanismClass : MechanismClass
};

void initHHGABAAMechanism(int cuda_init);

#endif

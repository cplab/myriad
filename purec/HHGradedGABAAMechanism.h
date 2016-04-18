/**
 * @file    HHGradedGABAAMechanism.h
 * 
 * @brief   Hodgkin-Huxley GABA-a Mechanism class definition file.
 *
 * @details Defines the Hodgkin-Huxley GABA-a Mechanism class specification for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    April 9, 2014
 */
#ifndef HHGABAACURRMECHANISM_H
#define HHGABAACURRMECHANISM_H

#include "MyriadObject.h"
#include "Mechanism.h"

// Generic pointers for new/class-of purposes

extern const void* HHGradedGABAAMechanism;
extern const void* HHGradedGABAAMechanismClass;

/**
 *   HHGradedGABAAMechanism mechanism for Hodgkin-Huxley GABA-a synapse.
 *
 * @see Mechanism
 */
struct HHGradedGABAAMechanism
{
    //! HHGradedGABAAMechanism : Mechanism
	struct Mechanism _;
    //! Synaptic gating variable (unitless, >=0, <=1)
	scalar g_s[SIMUL_LEN];
    //! Maximum synaptic conductance - nS
	scalar g_max;
    //! Half-activation potential - mV
	scalar theta;
    //! Maximal slope of the sigmoidal synaptic function
	scalar sigma;
    //! Channel opening time constant - ms
	scalar tau_alpha;
    //! Channel closing time constant - ms
	scalar tau_beta;
    //! Synaptic reversal potential - mV
	scalar gaba_rev;
};

struct HHGradedGABAAMechanismClass
{
	struct MechanismClass _; //! HHGradedGABAAMechanismClass : MechanismClass
};

void initHHGradedGABAAMechanism(const bool init_cuda);

#endif

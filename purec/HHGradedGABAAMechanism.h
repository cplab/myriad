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
	double g_s[SIMUL_LEN];
    //! Maximum synaptic conductance - nS
	double g_max;
    //! Half-activation potential - mV
	double theta;
    //! Maximal slope of the sigmoidal synaptic function
	double sigma;
    //! Channel opening time constant - ms
	double tau_alpha;
    //! Channel closing time constant - ms
	double tau_beta;
    //! Synaptic reversal potential - mV
	double gaba_rev;
};

struct HHGradedGABAAMechanismClass
{
	struct MechanismClass _; //! HHGradedGABAAMechanismClass : MechanismClass
};

extern void initHHGradedGABAAMechanism(void);

#endif

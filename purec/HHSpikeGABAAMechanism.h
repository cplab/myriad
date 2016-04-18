/**
   @file    HHSpikeGABAAMechanism.h
 
   @brief   Spike-Mediated Hodgkin-Huxley GABA-a Mechanism class definition file.
 
   @details Defines the Spike-Mediated Hodgkin-Huxley GABA-a Mechanism class specification for Myriad
 
   @author  Pedro Rittner
 
   @date    May 22, 2014
 */
#ifndef HHSPIKEGABAACURRMECHANISM_H
#define HHSPIKEGABAACURRMECHANISM_H

#include <stdbool.h>

#include "MyriadObject.h"
#include "Mechanism.h"

// Generic pointers for new/class-of purposes

extern const void* HHSpikeGABAAMechanism;
extern const void* HHSpikeGABAAMechanismClass;

/**
   HHSpikeGABAAMechanism mechanism for Hodgkin-Huxley GABA-a synapse.

   @see Mechanism
 */
struct HHSpikeGABAAMechanism
{
	struct Mechanism _;     //! HHSpikeGABAAMechanism : Mechanism
    scalar prev_vm_thresh;  //! Presynaptic membrane voltage 'threshold' for firing - mV
    scalar t_fired;         //! Last known firing time for presynaptic neuron - ms
	scalar g_max;			//! Maximum synaptic conductance - nS
    scalar norm_const;      //! Normalization constant (A-bar)
    scalar peak_cond_t;     //! Time of peak conductance (tp)
	scalar tau_alpha; 		//! Channel opening time constant - ms
	scalar tau_beta;		//! Channel closing time constant - ms
	scalar gaba_rev;		//! Synaptic reversal potential - mV
};

struct HHSpikeGABAAMechanismClass
{
	struct MechanismClass _; //! HHSpikeGABAAMechanismClass : MechanismClass
};

void initHHSpikeGABAAMechanism(void);

#endif /* HHSPIKEGABAACURRMECHANISM_H */

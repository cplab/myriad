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
    double prev_vm_thresh;  //! Presynaptic membrane voltage 'threshold' for firing - mV
    double t_fired;         //! Last known firing time for presynaptic neuron - ms
	double g_max;			//! Maximum synaptic conductance - nS
    double norm_const;      //! Normalization constant (A-bar)
    double peak_cond_t;     //! Time of peak conductance (tp)
	double tau_alpha; 		//! Channel opening time constant - ms
	double tau_beta;		//! Channel closing time constant - ms
	double gaba_rev;		//! Synaptic reversal potential - mV
};

struct HHSpikeGABAAMechanismClass
{
	struct MechanismClass _; //! HHSpikeGABAAMechanismClass : MechanismClass
};

void initHHSpikeGABAAMechanism(const bool cuda_init);

#endif /* HHSPIKEGABAACURRMECHANISM_H */

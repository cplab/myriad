/**
   @file    HHSpikeGABAAMechanism.h
 
   @brief   Spike-Mediated Hodgkin-Huxley GABA-a Mechanism class definition file.
 
   @details Defines the Spike-Mediated Hodgkin-Huxley GABA-a Mechanism class specification for Myriad
 
   @author  Pedro Rittner
 
   @date    May 22, 2014
 */
#ifndef HHSPIKEGABAACURRMECHANISM_H
#define HHSPIKEGABAACURRMECHANISM_H

#include "MyriadObject.hpp"
#include "Mechanism.hpp"

/**
   HHSpikeGABAAMechanism mechanism for Hodgkin-Huxley GABA-a synapse.

   @see Mechanism
 */
struct HHSpikeGABAAMechanism : public Mechanism
{
public:
    HHSpikeGABAAMechanism(uint64_t source_id,
                          double prev_vm_thresh,
                          double t_fired,
                          double g_max,
                          double tau_alpha,
                          double tau_beta,
                          double gaba_rev);
    
    virtual double mechanism_fxn(Compartment& post_comp,
                                 const double global_time,
                                 const uint64_t curr_step) override;

private:
    double prev_vm_thresh;  //! Presynaptic membrane voltage 'threshold' for firing - mV
    double t_fired;         //! Last known firing time for presynaptic neuron - ms
	double g_max;			//! Maximum synaptic conductance - nS
    double norm_const;      //! Normalization constant (A-bar)
    double peak_cond_t;     //! Time of peak conductance (tp)
	double tau_alpha; 		//! Channel opening time constant - ms
	double tau_beta;		//! Channel closing time constant - ms
	double gaba_rev;		//! Synaptic reversal potential - mV
};

#endif /* HHSPIKEGABAACURRMECHANISM_H */

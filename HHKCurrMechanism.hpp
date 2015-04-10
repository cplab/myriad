/**
 * @file    HHNaCurrMechanism.h
 *
 * @brief   Hodgkin-Huxley Sodium Mechanism class definition file.
 *
 * @details Defines the Hodgkin-Huxley Sodium Mechanism class specification for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    April 9, 2014
 */
#ifndef HHKCURRMECHANISM_H
#define HHKCURRMECHANISM_H

#include "MyriadObject.hpp"
#include "Mechanism.hpp"

/**
 * HHKCurrMechanism mechanism for Hodgkin-Huxley potassium channel.
 *
 * @see Mechanism
 */
class HHKCurrMechanism : public Mechanism
{
public:
    HHKCurrMechanism(uint64_t source_id, double g_k, double e_k, double hh_n);
    
    virtual double mechanism_fxn(Compartment& post_comp,
                                 const double global_time,
                                 const uint64_t curr_step) override;
private:
    //! Sodium channel conductance - nS
	double g_k;
    //! Sodium reversal potential - mV
	double e_k;
    //! @TODO Figure out what hh_n is actually called
	double hh_n;
};

#endif

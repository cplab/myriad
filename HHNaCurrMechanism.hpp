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
#ifndef HHNACURRMECHANISM_H
#define HHNACURRMECHANISM_H

#include "MyriadObject.hpp"
#include "Mechanism.hpp"

/**
 * HHNaCurrMechanism mechanism for Hodgkin-Huxley sodium channel.
 *
 * @see Mechanism
 */
class HHNaCurrMechanism : public Mechanism
{
public:
    HHNaCurrMechanism(uint64_t source_id,
                      double g_na,
                      double e_na,
                      double hh_m,
                      double hh_h);

    virtual double mechanism_fxn(const Compartment* pre_comp,
                                 const Compartment* post_comp,
                                 const double global_time,
                                 const uint64_t curr_step) override;
    
private:
    //! Sodium channel conductance - nS
	double g_na;
    //! Sodium reversal potential - mV
	double e_na;
    //! @TODO Figure out what hh_m is actually called
	double hh_m;
    //! @TODO Figure out what hh_h is actually called
	double hh_h;	    
};

#endif

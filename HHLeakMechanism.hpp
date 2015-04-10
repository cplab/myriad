/**
 * @file    HHLeakMechanism.h
 *
 * @brief   Hodgkin-Huxley Leak Mechanism class definition file.
 *
 * @details Defines the Hodgkin-Huxley Leak Mechanism class specification for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    April 9, 2014
 */
#ifndef HHLEAKMECHANISM_H
#define HHLEAKMECHANISM_H

#include "MyriadObject.hpp"
#include "Mechanism.hpp"

/**
 * HHLeakMechanism mechanism for Hodgkin-Huxley leak channel.
 *
 * @see Mechanism
 */
class HHLeakMechanism : public Mechanism
{
private:
    //! Leak Conductance - nS
	double g_leak;
    //! Leak Reversal Potential - mV
	double e_rev;
public:
    virtual double mechanism_fxn(Compartment& post_comp,
                                 const double global_time,
                                 const uint64_t curr_step) override;
    HHLeakMechanism(uint64_t source_id, double g_leak, double e_rev);
};

#endif /* HHLEAKMECHANISM_H */

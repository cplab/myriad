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

#include <stdbool.h>

#include "MyriadObject.h"
#include "Mechanism.h"

// Generic pointers for new(HHLeakMechanism) purposes
extern const void* HHLeakMechanism;
// Generic pointers for new(HHLeakMechanismClass) purposes
extern const void* HHLeakMechanismClass;

/**
 *   HHLeakMechanism mechanism for Hodgkin-Huxley leak channel.
 *
 * @see Mechanism
 */
struct HHLeakMechanism
{
    //! HHLeakMechanism : Mechanism
	struct Mechanism _;
    //! Leak Conductance - nS
	double g_leak;
    //! Leak Reversal Potential - mV
	double e_rev;       
};

struct HHLeakMechanismClass
{
    //! HHLeakMechanismClass : MechanismClass
	struct MechanismClass _;
};

extern void initHHLeakMechanism(void);

#endif /* HHLEAKMECHANISM_H */

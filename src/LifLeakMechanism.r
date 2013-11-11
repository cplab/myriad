/*
 * LifLeakMechanism.r
 *
 *  Created on: Nov 4, 2013
 *      Author: pedro
 */

#ifndef LIFLEAKMECHANISM_R_
#define LIFLEAKMECHANISM_R_

#include "Mechanism.r"

struct LifLeakMechanism
{
	const struct Mechanism _;
	double g_leak;	//! Leak channel conductance - nS
	double e_rev;	//! Reversal leak potential - mV
};

#endif /* LIFLEAKMECHANISM_R_ */

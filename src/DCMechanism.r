/*
 * DCMechanism.r
 *
 *  Created on: Nov 4, 2013
 *      Author: pedro
 */

#ifndef DCMECHANISM_R_
#define DCMECHANISM_R_

#include "Mechanism.r"

struct DCMechanism
{
	double amplitude;	//!	Pulse amplitude in nA
	double start;		//! Onset time of pulse in ms
	double stop;		//! End time of pulse in ms
};

#endif /* DCMECHANISM_R_ */

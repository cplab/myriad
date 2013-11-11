/*
 * LifCompartment.r
 *
 *  Created on: Oct 18, 2013
 *      Author: pedro
 */

#ifndef LIFCOMPARTMENT_R_
#define LIFCOMPARTMENT_R_

#include "Compartment.r"

struct LifCompartment
{
	const struct Compartment _;
	double v_rest;
	double cm;
	double tau_ref;
	double i_offset;
	double v_reset;
	double v_thresh;
	double t_fired;
	double* vm;
};

struct LifCompartmentClass
{
	const struct CompartmentClass _;			/* CompartmentClass : Class */
};

#endif /* LIFCOMPARTMENT_R_ */

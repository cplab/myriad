/*
 * Mechanism.r
 *
 *  Created on: Oct 18, 2013
 *      Author: pedro
 */

#ifndef MECHANISM_R_
#define MECHANISM_R_

//	Include Object representation
#include "Object.r"

#include "Mechanism.h"

struct Mechanism
{
	const struct Object _;			//! Point : Object
	unsigned int source_id;		//!	ID (or index) of "source" of mechanism
};

double super_mechanism_fxn (	const void* _class,
								void* _self,
								void* pre_comp,
								void* post_comp,
								const double dt,
								const double global_time,
								const unsigned int curr_step);

struct MechanismClass
{
	const struct Class _;			/* MechanismClass : Class */
	double (*mechanism_fxn) (	void* _self,
								void* pre_comp,
								void* post_comp,
								const double dt,
								const double global_time,
								const unsigned int curr_step);
};

#endif /* MECHANISM_R_ */

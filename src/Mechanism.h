/*
 * Mechanism.h
 *
 *  Created on: Oct 18, 2013
 *      Author: pedro
 */

#ifndef MECHANISM_H_
#define MECHANISM_H_

//	Import Object.h since this is an object
#include "Object.h"

//	Object definition

extern const void* Mechanism;		//	Declare for new() operator

double mechanism_fxn (void* _self,
						void* pre_comp,
						void* post_comp,
						const double dt,
						const double global_time,
						const unsigned int curr_step);

//	Class definition

extern const void* MechanismClass;	//	Declare for class definition

void initMechanism(void);

#endif /* MECHANISM_H_ */

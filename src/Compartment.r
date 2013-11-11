/*
 * Compartment.r
 *
 *  Created on: Oct 17, 2013
 *      Author: pedro
 */

#ifndef	COMPARTMENT_R
#define	COMPARTMENT_R

//	Include Object representation
#include "Object.r"
#include "Mechanism.r"

struct Compartment
{
	const struct Object _;			//! Point : Object
	unsigned int my_id;			//!	ID (or index) of this compartment
	unsigned int mech_count;		//! Number of mechanisms ending at this compartment
	struct Mechanism** mechanisms;	//!	Channels 'ending' at this compartment
};

void super_step_simulate_fxn (const void * _class,
							  	 void* _self,
							  	 void** network,
							  	 const double dt,
							  	 const double global_time,
							  	 const unsigned int curr_step);

int super_add_mechanism (const void* _class,
					 	 	 void* _self,
					 	 	 void* mechanism);

int super_remove_mechanism(const void* _class,
								void* _self,
								void* mechanism);

struct CompartmentClass
{
	const struct Class _;			/* CompartmentClass : Class */
	void (*step_simulate_fxn) (void* self,
							   void** network,
							   const double dt,
							   const double global_time,
							   const unsigned int curr_step);
	int (*add_mechanism) (void* self,
						  void* mechanism);
	int (*remove_mechanism) (void* self,
							 void* mechanism);
};

#endif

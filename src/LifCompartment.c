/*
 * LifCompartment.c
 *
 *  Created on: Nov 3, 2013
 *      Author: pedro
 */

#include <assert.h>

#include "Compartment.r"
#include "Mechanism.r"

#include "LifCompartment.h"
#include "LifCompartment.r"

/*
 *	Compartment
 */

static void* LifCompartment_ctor(void* _self, va_list * app)
{
	struct LifCompartment * self = super_ctor(LifCompartment, _self, app);

	self->vm		= (double*) calloc(va_arg(*app, unsigned int), sizeof(double));
	self->v_rest 	= va_arg(*app, double);
	self->vm[0] = self->v_rest;
	self->cm 		= va_arg(*app, double);
	self->tau_ref 	= va_arg(*app, double);
	self->i_offset 	= va_arg(*app, double);
	self->v_reset 	= va_arg(*app, double);
	self->v_thresh 	= va_arg(*app, double);
	self->t_fired 	= va_arg(*app, double);

	return self;
}

static void LifCompartment_step_simulate_fxn (void* _self,
													void** network,
		  	  	  	  	  	  	  	  	       	    const double dt,
		  	  	  	  	  	  	  	  	       	    const double global_time,
		  	  	  	  	  	  	  	  	       	    const unsigned int curr_step)
{
	struct LifCompartment * self = _self;

	double I_sum = 0.0;

	//	If still in refractory period, ignore incoming currents
	if (self->t_fired > 0.0 && global_time < self->t_fired + self->tau_ref)
	{
		self->vm[curr_step] = self->v_reset;
		return;
	}

	//	Calculate mechanism contribution to current term
	for (unsigned int i = 0; i < self->_.mech_count; i++)
	{
		struct Mechanism* curr_mech = self->_.mechanisms[i];
		struct Compartment* pre_comp = (struct Compartment*)(network[i]);

		if (isOf(curr_mech, Mechanism))
		{
			I_sum += mechanism_fxn(curr_mech, pre_comp, self, dt, global_time, curr_step);
		}
	}

	//	Calculate dVm, new membrane voltage
	const double dVm = dt * (I_sum + self->i_offset) / self->cm;
	const double prev_vm = self->vm[curr_step-1];
	const double new_vm = dVm + prev_vm;

	//	Go into refactory period if exceeded threshold, otherwise just set and return
	if (new_vm > self->v_thresh)
	{
		self->vm[curr_step] = self->v_rest;
		self->t_fired = global_time;
	}else{
		self->vm[curr_step] = new_vm;
	}

	printf("timestep: %u \t\tvm: %f\n",curr_step,self->vm[curr_step]);

	return;
}

/*
 *	LifCompartmentClass
 */

static void* LifCompartmentClass_ctor(void* _self, va_list * app)
{
	struct LifCompartmentClass * self = super_ctor(LifCompartmentClass, _self, app);

	return self;
}

/*
 *	initialization
 */

const void* LifCompartment, *LifCompartmentClass;

void initLifCompartment(void)
{
	if (!LifCompartmentClass)
	{
		initCompartment();
		LifCompartmentClass = new(CompartmentClass, "LifCompartmentClass",
								  CompartmentClass, sizeof(struct LifCompartmentClass),
								  ctor, LifCompartmentClass_ctor,
								  0);
	}

	if (!LifCompartment)
	{
		initCompartment();
		LifCompartment = new(LifCompartmentClass, "LifCompartment",
							 Compartment, sizeof(struct LifCompartment),
							 ctor, LifCompartment_ctor,
							 step_simulate_fxn, LifCompartment_step_simulate_fxn,
							 0);
	}

}


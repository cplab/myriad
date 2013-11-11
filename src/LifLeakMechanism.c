/*
 * LifLeakMechanism.c
 *
 *  Created on: Nov 4, 2013
 *      Author: pedro
 */

#include <assert.h>

#include "LifCompartment.r"

#include "LifLeakMechanism.h"
#include "LifLeakMechanism.r"

/*
 *	LifLeakMechanism
 */

static void* LifLeakMechanism_ctor(void* _self, va_list * app)
{
	struct LifLeakMechanism * self = super_ctor(LifLeakMechanism, _self, app);

	self->g_leak	= va_arg(*app, double);
	self->e_rev		= va_arg(*app, double);

	return self;
}

static double LifLeakMechanism_mechanism_fxn(void* _self,
												void* pre_comp,
												void* post_comp,
												const double dt,
												const double global_time,
												const unsigned int curr_step)
{
	const struct LifLeakMechanism * self = _self;

	// TODO: Check for appropriate compartment type in LifLeakMechanism by implementing LifCompartmentClass
	if (isOf(pre_comp , LifCompartmentClass))
	{
		printf("lif_leak_calc: cannot calculate leak current for non-LIF compartment.\n");
		return 0.0;
	}

	struct LifCompartment* my_lif_comp = (struct LifCompartment*) pre_comp;
	const double vm = my_lif_comp->vm[curr_step-1];

	return -self->g_leak * (vm - self->e_rev);
}

/*
 *	initialization
 */

const void *LifLeakMechanism;

void initLifLeakMechanism(void)
{
	if (!LifLeakMechanism)
	{
		initMechanism();
		LifLeakMechanism = new(MechanismClass,	"LifLeakMechanism",
							Mechanism, sizeof(struct LifLeakMechanism),
							ctor, LifLeakMechanism_ctor,
							mechanism_fxn, LifLeakMechanism_mechanism_fxn,
							0);
	}
}

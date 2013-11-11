/*
 * DCMechanism.c
 *
 *  Created on: Nov 4, 2013
 *      Author: pedro
 */

#include <assert.h>

#include "DCMechanism.h"
#include "DCMechanism.r"

/*
 *	DCMechanism
 */

static void* DCMechanism_ctor(void* _self, va_list * app)
{
	struct DCMechanism * self = super_ctor(DCMechanism, _self, app);

	self->amplitude	= va_arg(*app, double);
	self->start		= va_arg(*app, double);
	self->stop 		= va_arg(*app, double);

	return self;
}

static double DCMechanism_mechanism_fxn(void* _self,
											void* pre_comp,
											void* post_comp,
											const double dt,
											const double global_time,
											const unsigned int curr_step)
{
	const struct DCMechanism * self = _self;

	return (global_time >= self->start && global_time <= self->stop) ? self->amplitude : 0.0;
}

/*
 *	initialization
 */

const void *DCMechanism;

void initDCMechanism(void)
{
	if (!DCMechanism)
	{
		initMechanism();
		DCMechanism = new(MechanismClass,	"DCMechanism",
						Mechanism, sizeof(struct DCMechanism),
						ctor, DCMechanism_ctor,
						mechanism_fxn, DCMechanism_mechanism_fxn,
						0);
	}
}

/*
 * Mechanism.c
 *
 *  Created on: Oct 18, 2013
 *      Author: pedro
 */

#include <assert.h>

#include "Mechanism.h"
#include "Mechanism.r"

/*
 *	Mechanism
 */

static void* Mechanism_ctor(void* _self, va_list * app)
{
	struct Mechanism * self = super_ctor(Mechanism, _self, app);

	self->source_id	= va_arg(*app, unsigned int);

	return self;
}

static void Mechanism_mechanism_fxn(	void* _self,
										void* pre_comp,
										void* post_comp,
										const double dt,
										const double global_time,
										const unsigned int curr_step)
{
	const struct Mechanism * self = _self;

	printf("My source id is %u\n", self->source_id);
}

double mechanism_fxn (void* _self,
						void* pre_comp,
						void* post_comp,
						const double dt,
						const double global_time,
						const unsigned int curr_step)
{
	const struct MechanismClass * class = classOf(_self);

	assert(class->mechanism_fxn);
	return class->mechanism_fxn(_self,
								pre_comp,
								post_comp,
								dt,
								global_time,
								curr_step);
}

double super_mechanism_fxn (	const void* _class,
								void* _self,
								void* pre_comp,
								void* post_comp,
								const double dt,
								const double global_time,
								const unsigned int curr_step)
{
	const struct MechanismClass * superclass = super(_class);

	assert(_self && superclass->mechanism_fxn);
	return superclass->mechanism_fxn(	_self,
										pre_comp,
										post_comp,
										dt,
										global_time,
										curr_step);
}

/*
 *	MechanismClass
 */

static void* MechanismClass_ctor(void* _self, va_list * app)
{
	struct MechanismClass * self = super_ctor(MechanismClass, _self, app);
	voidf selector;

	while ((selector = va_arg(*app, voidf)))
	{
		voidf method = va_arg(*app, voidf);

		if (selector == (voidf) mechanism_fxn)
		{
			*(voidf *) &self->mechanism_fxn = method;
		}
	}

	return self;
}

/*
 *	initialization
 */

const void* MechanismClass, *Mechanism;

void initMechanism(void)
{
	if (!MechanismClass)
	{
		MechanismClass = new(Class,
							"MechanismClass",
							Class,
							sizeof(struct MechanismClass),
							ctor, MechanismClass_ctor,
							0);
	}

	if (!Mechanism)
	{
		Mechanism = new(MechanismClass,
						"Mechanism",
						Object,
						sizeof(struct Mechanism),
						ctor, Mechanism_ctor,
						mechanism_fxn, Mechanism_mechanism_fxn,
						0);
	}
}


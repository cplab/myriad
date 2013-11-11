/*
 * Compartment.c
 *
 *  Created on: Oct 17, 2013
 *      Author: pedro
 */

#include <assert.h>

#include "Compartment.h"
#include "Compartment.r"

/*
 *	Compartment
 */

static void* Compartment_ctor(void* _self, va_list * app)
{
	struct Compartment * self = super_ctor(Compartment, _self, app);

	self->my_id 		= va_arg(*app, unsigned int);
	self->mech_count 	= va_arg(*app, unsigned int);
	self->mechanisms 	= (struct Mechanism**) va_arg(*app, void**);

	return self;
}

/*
 * Step simulate function
 */

static void Compartment_step_simulate_fxn (void* _self,
												void** network,
		  	  	  	  	  	  	  	  	       const double dt,
		  	  	  	  	  	  	  	  	       const double global_time,
		  	  	  	  	  	  	  	  	       const unsigned int curr_step)
{
	const struct Compartment * self = _self;

	printf("My id is %u and I have %u mechanisms\n", self->my_id, self->mech_count);
}

void step_simulate_fxn (void* _self,
						  void** network,
						  const double dt,
						  const double global_time,
						  const unsigned int curr_step)
{
	const struct CompartmentClass * class = classOf(_self);

	assert(class->step_simulate_fxn);
	class->step_simulate_fxn(_self,
							network,
							dt,
							global_time,
							curr_step);
}

void super_step_simulate_fxn (const void * _class,
							  	 void* _self,
							  	 void** network,
							  	 const double dt,
							  	 const double global_time,
							  	 const unsigned int curr_step)
{
	const struct CompartmentClass * superclass = super(_class);

	assert(_self && superclass->step_simulate_fxn);
	superclass->step_simulate_fxn(_self,
								 network,
								 dt,
								 global_time,
								 curr_step);
}

/*
 * Add mechanism function
 */

static int Compartment_add_mechanism (void* _self,
										    void* mechanism)
{
	struct Compartment* self = _self;

	assert(mechanism && "Compartment: cannot add a NULL mechanism.");
	//TODO: Find some way to assert something is a class of something else
//	assert(classOf(mechanism) == MechanismClass && "Cannot add non-mechanism.");

	struct Mechanism* mech = (struct Mechanism*) mechanism;

	if (self->mech_count == 0 || !self->mechanisms)
	{
		self->mechanisms = (struct Mechanism**) calloc(1, sizeof(struct Mechanism*));
		assert(self->mechanisms && "Failed to allocate mechanisms.");
		self->mech_count = 1;
	} else {
		self->mech_count += 1;
		self->mechanisms = (struct Mechanism**) realloc(self->mechanisms, self->mech_count * sizeof(struct Mechanism*));
		assert(self->mechanisms && "Failed to reallocate mechanisms.");
	}

	self->mechanisms[self->mech_count-1] = mech;

	return EXIT_SUCCESS;
}

int add_mechanism (void* _self,
					  void* mechanism)
{
	const struct CompartmentClass * class = classOf(_self);

	assert(class->add_mechanism);
	return class->add_mechanism(_self,
								mechanism);
}

int super_add_mechanism (const void * _class,
							 void* _self,
							 void* mechanism)
{
	const struct CompartmentClass * superclass = super(_class);

	assert(_self && superclass->add_mechanism);
	return superclass->add_mechanism(_self,
							  	  	  mechanism);
}

/*
 *	CompartmentClass
 */

static void* CompartmentClass_ctor(void* _self, va_list * app)
{
	struct CompartmentClass * self = super_ctor(CompartmentClass, _self, app);
	voidf selector;

	while ((selector = va_arg(*app, voidf)))
	{
		voidf method = va_arg(*app, voidf);

		if (selector == (voidf) step_simulate_fxn)
		{
			*(voidf *) &self->step_simulate_fxn = method;
		} else if (selector == (voidf) add_mechanism) {
			*(voidf *) &self->add_mechanism = method;
		}
//		else if (selector == (voidf) remove_mechanism) {
//			*(voidf *) &self->remove_mechanism = method;
//		}
	}

	return self;
}

/*
 *	initialization
 */

const void* CompartmentClass, *Compartment;

void initCompartment(void)
{
	if (!CompartmentClass)
	{
		CompartmentClass = new(Class,"CompartmentClass",
							   Class,sizeof(struct CompartmentClass),
							   ctor, CompartmentClass_ctor,
							   0);
	}

	if (!Compartment)
	{
		Compartment = new(CompartmentClass, "Compartment",
					Object, sizeof(struct Compartment),
					ctor, Compartment_ctor,
					step_simulate_fxn, Compartment_step_simulate_fxn,
					add_mechanism, Compartment_add_mechanism,
					//TODO: Add Compartment_remove_mechanism here
					0);
	}
}


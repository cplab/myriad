#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "Compartment.h"
#include "LifCompartment.h"
#include "DCMechanism.h"
#include "LifLeakMechanism.h"

int main (int argc, char ** argv)
{
	// Test Compartment
//	void * c;
//
//	initCompartment();
//
//	c = new(Compartment, 0, 0, NULL);
//
//	step_simulate_fxn(c, NULL, 0.0, 0.0, 0);
//
//	delete(c);

	// Init
	initLifCompartment();
	initLifLeakMechanism();
	initDCMechanism();

	// Global params
	const unsigned num_timesteps = 4000;
	const double dt = 0.01;
	const double global_time = 0.0;

	void** network = (void**) calloc(1, sizeof(void*));

	// Test LIF Compartment
	void* lif_c;

	double v_rest = -70.0;
	double cm = 1.0;
	double tau_ref = 5.0;
	double i_offset = 0.0;
	double v_r = -70.0;
	double v_t = -50.0;
	double t_f = -1.0;

	lif_c = new(LifCompartment,													// Class name
				0, 0, NULL, 													// Compartment params
				num_timesteps, v_rest, cm,	tau_ref, i_offset, v_r, v_t, t_f	// LIFCompartment params
				);

	network[0] = lif_c;

	// Test LIF Leak Current
	void* lif_c_leak;

	double g_leak = 0.10;
	double e_rev = -70.0;

	lif_c_leak = new(LifLeakMechanism,		// Class Name
					0,			// Mechanism params
					g_leak, e_rev);

	// Add LIF Leak Current to LIF Compartment
	add_mechanism(lif_c, lif_c_leak);

	for (unsigned int i = 1; i < num_timesteps; i++)
	{
		step_simulate_fxn(lif_c, network, dt, global_time, i);
	}

	delete(lif_c); 	//TODO: Implement delete methods for LifCompartment to actually delete arrays

	puts("Compartment test completed successfully.");

	return 0;
}

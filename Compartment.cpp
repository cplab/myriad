#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>

#include "MyriadObject.hpp"
#include "Compartment.hpp"

/////////////////////////////////
// Compartment Super Overrides //
/////////////////////////////////

Compartment::Compartment(uint64_t id, uint64_t num_mechs)
{
	this->id = id;
	this->num_mechs = num_mechs;
}

// Simulate function
void Compartment::simul_fxn(Compartment network[],
                            const double global_time,
                            const uint64_t curr_step)
{
	printf("My id is %lu\n", this->id);
	printf("My num_mechs is %lu\n", this->num_mechs);
	return;
}

// Add mechanism function
int Compartment::add_mech(Mechanism& mechanism)
{
    if (this->num_mechs + 1 >= MAX_NUM_MECHS)
    {
        fprintf(stderr, "Cannot add mechanism to Compartment: out of room.\n");
        return -1;
    }
	
	this->num_mechs++;
	this->my_mechs[this->num_mechs-1] = mechanism;

	return 0;
}

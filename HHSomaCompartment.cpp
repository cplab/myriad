#include "MyriadObject.hpp"
#include "HHSomaCompartment.hpp"

HHSomaCompartment::HHSomaCompartment(uint64_t id,
                                     uint64_t num_mechs,
                                     const double cm,
                                     const double init_vm) :
    Compartment(id, num_mechs)
{
    this->cm = cm;

    this->vm.reserve(SIMUL_LEN);
    this->vm[0] = init_vm;
}

void HHSomaCompartment::simul_fxn(Compartment network[],
                                  const double global_time,
                                  const uint64_t curr_step)
{
	double I_sum = 0.0;

	//	Calculate mechanism contribution to current term
#pragma GCC ivdep
	for (uint64_t i = 0; i < this->num_mechs; i++)
	{
		Mechanism& curr_mech = this->my_mechs[i]; // TODO: GENERICSE DIS
		Compartment& pre_comp = network[curr_mech.source_id];

		//TODO: Make this conditional on specific Mechanism types
		//if (curr_mech->fx_type == CURRENT_FXN)
		I_sum += curr_mech.mechanism_fxn(pre_comp, global_time, curr_step);
	}

	//	Calculate new membrane voltage: (dVm) + prev_vm
	this->vm[curr_step] = (DT * (I_sum) / (this->cm)) + this->vm[curr_step - 1];

	return;
}

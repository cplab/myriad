#include <cstdlib>
#include <cmath>

#include "MyriadObject.hpp"
#include "Mechanism.hpp"
#include "HHSomaCompartment.hpp"
#include "HHNaCurrMechanism.hpp"

///////////////////////////////////////
// HHNaCurrMechanism Super Overrides //
///////////////////////////////////////

HHNaCurrMechanism::HHNaCurrMechanism(uint64_t source_id,
                                     double g_na,
                                     double e_na,
                                     double hh_m,
                                     double hh_h) : Mechanism(source_id)
{
	this->g_na = g_na;
	this->e_na = e_na;
	this->hh_m = hh_m;
	this->hh_h = hh_h;
}

double HHNaCurrMechanism::mechanism_fxn(const Compartment* pre_comp,
                                        const Compartment* post_comp,
                                        const double global_time,
                                        const uint64_t curr_step)
{
	const HHSomaCompartment* c2 = static_cast<const HHSomaCompartment*>(post_comp);

	//	Channel dynamics calculation
	const double pre_vm = c2->vm[curr_step-1];
    
	const double alpha_m = (-0.1 * (pre_vm + 35.)) / (EXP(-0.1 * (pre_vm + 35.)) - 1.);
	const double beta_m =  4. * EXP((pre_vm + 60.) / -18.);
	const double alpha_h = (0.128) / (EXP((pre_vm + 41.)/ 18.));
	const double beta_h = 4. / (1. + EXP(-(pre_vm + 18.)/ 5.));

	const double minf = (alpha_m/(alpha_m + beta_m));
	this->hh_h += DT * 5. *(alpha_h * (1. - this->hh_h) - (beta_h * this->hh_h));

    //	I = g_Na * minf^3 * hh_h * (Vm[t-1] - e_rev)
    const double I_Na = -this->g_na * minf * minf * minf * this->hh_h *
        (pre_vm - this->e_na);
    return I_Na;
}

#include <cstdlib>
#include <cmath>

#include "MyriadObject.hpp"
#include "Mechanism.hpp"
#include "HHSomaCompartment.hpp"
#include "HHKCurrMechanism.hpp"

///////////////////////////////////////
// HHKCurrMechanism Super Overrides //
///////////////////////////////////////

HHKCurrMechanism::HHKCurrMechanism(uint64_t source_id,
                                   double g_k,
                                   double e_k,
                                   double hh_n) : Mechanism(source_id)
{
	this->g_k = g_k;
	this->e_k = e_k;
	this->hh_n = hh_n;
}

double HHKCurrMechanism::mechanism_fxn(Compartment& post_comp,
                                       const double global_time,
                                       const uint64_t curr_step)
{
	const HHSomaCompartment c2 = (const HHSomaCompartment&) post_comp;

	//	Channel dynamics calculation
	const double pre_vm = c2.vm[curr_step-1];

    const double alpha_n = (-0.01 * (pre_vm + 34.)) / (EXP((pre_vm+34.0)/-1.) - 1.);
    const double beta_n  = 0.125 * EXP((pre_vm + 44.)/-80.);

    this->hh_n += DT*5.*(alpha_n*(1.-this->hh_n) - beta_n*this->hh_n);

    //	I_K = g_K * hh_n^4 * (Vm[t-1] - e_K)
    return -this->g_k * this->hh_n * this->hh_n * this->hh_n *
        this->hh_n * (pre_vm - this->e_k);
}

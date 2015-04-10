#include "MyriadObject.hpp"
#include "Mechanism.hpp"
#include "HHSomaCompartment.hpp"
#include "HHLeakMechanism.hpp"

/////////////////////////////////////
// HHLeakMechanism Super Overrides //
/////////////////////////////////////

HHLeakMechanism::HHLeakMechanism(uint64_t source_id,
                                 double g_leak,
                                 double e_rev) :
    Mechanism(source_id)
{
	this->g_leak = g_leak;
    this->e_rev = e_rev;
}

double HHLeakMechanism::mechanism_fxn(Compartment& post_comp,
                                      const double global_time,
                                      const uint64_t curr_step)
{
	const HHSomaCompartment c2 = (const HHSomaCompartment&) post_comp;

	//	No extracellular compartment. Current simply "disappears".
    return -this->g_leak * (c2.vm[curr_step-1] - this->e_rev);
}


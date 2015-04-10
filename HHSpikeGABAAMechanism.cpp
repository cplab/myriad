#include <cstdlib>
#include <cmath>

#include "MyriadObject.hpp"
#include "Mechanism.hpp"
#include "HHSomaCompartment.hpp"
#include "HHSpikeGABAAMechanism.hpp"

///////////////////////////////////////
// HHSpikeGABAAMechanism Super Overrides //
///////////////////////////////////////

HHSpikeGABAAMechanism::HHSpikeGABAAMechanism(uint64_t source_id,
                                             double prev_vm_thresh,
                                             double t_fired,
                                             double g_max,
                                             double tau_alpha,
                                             double tau_beta,
                                             double gaba_rev) :
    Mechanism(source_id)
{
    this->prev_vm_thresh = prev_vm_thresh;
    this->t_fired = t_fired;
	this->g_max = g_max;
	this->tau_alpha = tau_alpha;
	this->tau_beta = tau_beta;
	this->gaba_rev = gaba_rev;

    // Automatic calculations for t_p and A-bar, from Guoshi
    this->peak_cond_t = ((this->tau_alpha * this->tau_beta) /
        (this->tau_beta - this->tau_alpha)) * 
        log(this->tau_beta / this->tau_alpha);

    this->norm_const = 1.0 / 
        (exp(-this->peak_cond_t/this->tau_beta) - 
         exp(-this->peak_cond_t/this->tau_alpha));
}

double HHSpikeGABAAMechanism::mechanism_fxn(Compartment& post_comp,
                                 const double global_time,
                                 const uint64_t curr_step)
{
	const HHSomaCompartment c2 = (const HHSomaCompartment&) post_comp;

	//	Channel dynamics calculation
    const double pre_pre_vm = (curr_step > 1) ? c2.vm[curr_step-2] : INFINITY;
	const double pre_vm = c1->vm[curr_step-1];
	const double post_vm = c2->vm[curr_step-1];
    
    // If we just fired
    if (pre_vm > this->prev_vm_thresh && pre_pre_vm < this->prev_vm_thresh)
    {
        this->t_fired = global_time;
    }

    if (this->t_fired != -INFINITY)
    {
        const double g_s = exp(-(global_time - this->t_fired) / this->tau_beta) - 
            exp(-(global_time - this->t_fired) / this->tau_alpha);
        const double I_GABA = this->norm_const * -this->g_max * g_s * (post_vm - this->gaba_rev);
        return I_GABA;        
    } else {
        return 0.0;
    }
}

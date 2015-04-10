#ifndef HHSOMACOMPARTMENT_H
#define HHSOMACOMPARTMENT_H

#include <stdint.h>
#include <vector>

#include "Compartment.hpp"

class HHSomaCompartment : public Compartment
{
public:
    HHSomaCompartment(uint64_t id,
                      uint64_t num_mechs,
                      const double cm,
                      const double init_vm);
    virtual void simul_fxn(Compartment network[],
                           const double global_time,
                           const uint64_t curr_step) override;
    //! Membrane voltage - mV
    std::vector<double> vm;
    //! Capacitance - nF
    double cm;
};

#endif

#ifndef HHSOMACOMPARTMENT_H
#define HHSOMACOMPARTMENT_H

#include <stdbool.h>
#include <stdint.h>

#include "Compartment.h"

extern const void* HHSomaCompartment;
extern const void* HHSomaCompartmentClass;

struct HHSomaCompartment
{
    //! HHSomaCompartment : Compartment
    struct Compartment _;
    //! Membrane voltage - mV
    double vm[SIMUL_LEN];
    //! Length of soma_vm array
    uint_fast32_t vm_len;
    //! Capacitance - nF
    double cm;   
};

struct HHSomaCompartmentClass
{
    //! HHSomaCompartmentClass : CompartmentClass
    struct CompartmentClass _;
};

extern void initHHSomaCompartment(void);

#endif

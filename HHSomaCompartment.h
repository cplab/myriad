#ifndef HHSOMACOMPARTMENT_H
#define HHSOMACOMPARTMENT_H

#include "Compartment.h"

extern const void* HHSomaCompartment;
extern const void* HHSomaCompartmentClass;

struct HHSomaCompartment
{
    struct Compartment _; //! HHSomaCompartment : Compartment
    double* vm;           //! Membrane voltage - mV
    unsigned int vm_len;  //! Length of soma_vm array
    double cm;            //! Capacitance - nF
};

struct HHSomaCompartmentClass
{
    struct CompartmentClass _; //! HHSomaCompartmentClass : CompartmentClass
};

extern void initHHSomaCompartment(int init_cuda);

#endif

#ifndef HHSOMACOMPARTMENT_H
#define HHSOMACOMPARTMENT_H

#include "Compartment.h"

#include "HHSomaCompartment_meta.h"

extern const void* HHSOMACOMPARTMENT_OBJECT;
extern const void* HHSOMACOMPARTMENT_CLASS;

struct HHSOMACOMPARTMENT_OBJECT
{
	struct COMPARTMENT_OBJECT SUPERCLASS_HHSOMACOMPARTMENT_OBJECT_NAME;                // HHSomaCompartment : Compartment
	double* HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE;               // Membrane voltage - mV
	unsigned int HHSOMACOMPARTMENT_MEMBRANE_VOLTAGE_LENGTH;   // Length of soma_vm array
	double HHSOMACOMPARTMENT_CAPACITANCE;                     // Capacitance - nF
};

struct HHSOMACOMPARTMENT_CLASS
{
	struct COMPARTMENT_CLASS SUPERCLASS_HHSOMACOMPARTMENT_CLASS_NAME; // HHSomaCompartmentClass : CompartmentClass
};

extern void initHHSomaCompartment(int init_cuda);

#endif

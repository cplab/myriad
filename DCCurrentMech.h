/**
   @file    DCCurrentMech.h
 
   @brief   DC Current Mechanism definition file.
 
   @details Defines the DCCurrentMech class specification for Myriad
 
   @author  Pedro Rittner
 
   @date    May 5, 2014
 */
#ifndef DCCURRENTMECH_H
#define DCCURRENTMECH_H

#include "MyriadObject.h"
#include "Mechanism.h"

// Generic pointers for new/class-of purposes

extern const void* DCCurrentMech;
extern const void* DCCurrentMechClass;

// -----------------------------------------

/**
   DC Current Mechanism object structure definition.

   Stores DC Current Mechanism state.

   @see Mechanism
 */
struct DCCurrentMech
{
	const struct Mechanism _; //! DCCurrentMech : Mechanism
	unsigned int t_start;     //! Time step at which current starts flowing
	unsigned int t_stop;      //! Time step at which current stops flowing
	double amplitude;         //! Current amplitude in nA
};

/**
   DC Current Mechanism class structure definition.

   Defines DC Current Mechanism behavior.

   @see MechanismClass
 */
struct DCCurrentMechClass
{
	struct MechanismClass _; //! MechanismClass : MyriadClass
};

// -------------------------------------

/**
   Initializes prototype DC Current Mechanism infrastructure on the heap.

   @param[in]    init_cuda    flag for directing CUDA protoype initialization
 */
extern void initDCCurrMech(const int init_cuda);

#endif

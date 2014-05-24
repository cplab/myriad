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

#include "DCCurrentMech_meta.h"

// Generic pointers for new/class-of purposes

extern const void* DCCURRENTMECHANISM_OBJECT;
extern const void* DCCURRENTMECHANISM_CLASS;

// -----------------------------------------

/**
   DC Current Mechanism object structure definition.

   Stores DC Current Mechanism state.

   @see Mechanism
 */
struct DCCURRENTMECHANISM_OBJECT
{
	const struct Mechanism SUPERCLASS_DCCURRENTMECHANISM_OBJECT_NAME; //! DCCurrentMech : Mechanism
	unsigned int DCCURRENTMECHANISM_T_START;     //! Time step at which current starts flowing
	unsigned int DCCURRENTMECHANISM_T_STOP;      //! Time step at which current stops flowing
	double DCCURRENTMECHANISM_AMPLITUDE;         //! Current amplitude in nA
};

/**
   DC Current Mechanism class structure definition.

   Defines DC Current Mechanism behavior.

   @see MechanismClass
 */
struct DCCURRENTMECHANISM_CLASS
{
	struct MechanismClass SUPERCLASS_DCCURRENTMECHANISM_CLASS_NAME; //! MechanismClass : MyriadClass
};

// -------------------------------------

/**
   Initializes prototype DC Current Mechanism infrastructure on the heap.

   @param[in]    init_cuda    flag for directing CUDA protoype initialization
 */
extern void initDCCurrMech(const int init_cuda);

#endif

/**
   @file    HHNaCurrMechanism.h
 
   @brief   Hodgkin-Huxley Sodium Mechanism class definition file.
 
   @details Defines the Hodgkin-Huxley Sodium Mechanism class specification for Myriad
 
   @author  Pedro Rittner
 
   @date    April 9, 2014
 */
#ifndef HHNACURRMECHANISM_H
#define HHNACURRMECHANISM_H

#include "MyriadObject.h"
#include "Mechanism.h"

#include "HHNaCurrMechanism_meta.h"

// Generic pointers for new/class-of purposes

extern const void* HHNACURRMECHANISM_OBJECT;
extern const void* HHNACURRMECHANISM_CLASS;

/**
   HHNaCurrMechanism mechanism for Hodgkin-Huxley sodium channel.

   @see Mechanism
 */
struct HHNACURRMECHANISM_OBJECT
{
	struct MECHANISM_OBJECT SUPERCLASS_HHNACURRMECHANISM_OBJECT_NAME; //! HHNaCurrMechanism : Mechanism
	double HHNACURRMECHANISM_CHANNEL_CONDUCTANCE;	    //! Sodium channel conductance - nS
	double HHNACURRMECHANISM_REVERSAL_POTENTIAL;	    //! Sodium reversal potential - mV
	double HHNACURRMECHANISM_HH_M;	    //! @TODO Figure out what hh_m is actually called
	double HHNACURRMECHANISM_HH_H;	    //! @TODO Figure out what hh_h is actually called
};

struct HHNACURRMECHANISM_CLASS
{
	struct MECHANISM_CLASS SUPERCLASS_HHNACURRMECHANISM_CLASS_NAME; //! HHNaCurrMechanismClass : MechanismClass
};

MYRIAD_FXN_METHOD_HEADER_GEN_NO_SUFFIX(DYNAMIC_INIT_FXN_RET, DYNAMIC_INIT_FXN_ARGS, HHNACURRMECHANISM_INIT_FXN_NAME);
//void initHHNaCurrMechanism(int cuda_init);

#endif

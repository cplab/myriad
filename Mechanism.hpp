/**
 * @file    Mechanism.h
 *
 * @brief   Generic Mechanism class definition file.
 *
 * @details Defines the generic Mechanism class specification for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    April 7, 2014
 */
#ifndef MECHANISM_H
#define MECHANISM_H

#include <stdint.h>

#include "MyriadObject.hpp"

class Compartment;

//! Mechanism function typedef
class Mechanism : public MyriadObject
{
public:
    virtual double mechanism_fxn(const Compartment* pre_comp,
                                 const Compartment* post_comp,
                                 const double global_time,
                                 const uint64_t curr_step);
    
    Mechanism(uint64_t source_id);
    Mechanism();
    
    //! Source ID of the pre-mechanism compartment
    uint64_t source_id;
};


#endif /* MECHANISM_H */

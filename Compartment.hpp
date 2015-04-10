/**
 * @file    Compartment.h
 *
 * @brief   Generic Compartment class definition file.
 *
 * @details Defines the Compartment class specification for Myriad
 *
 * @author  Pedro Rittner
 *
 * @date    April 7, 2014
 */
#ifndef COMPARTMENT_H
#define COMPARTMENT_H

#include <stdint.h>

#include "MyriadObject.hpp"
#include "Mechanism.hpp"


class Compartment : public MyriadObject
{
public:
    /**
     * Generic simulation function.
     *
     * @param[in]  network      list of pointers to other compartments in network
     * @param[in]  global_time  current global time of the simulation
     * @param[in]  curr_step    current global time step of the simulation
     */
    virtual void simul_fxn(Compartment network[],
                           const double global_time,
                           const uint64_t curr_step);

    //! Method for adding mechanisms to a compartment
    virtual int add_mech(Mechanism& mechanism);

    Compartment(uint64_t id, uint64_t num_mechs);

protected:
    //! This compartment's unique ID number
	uint64_t id;
    //! Number of mechanisms in this compartment
	uint64_t num_mechs;
    //! List of mechanisms in this compartment
	Mechanism my_mechs[MAX_NUM_MECHS];
};

#endif /* COMPARTMENT_H */

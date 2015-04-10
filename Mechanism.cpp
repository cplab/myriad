#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <cassert>
#include <cstring>

#include "MyriadObject.hpp"
#include "Mechanism.hpp"
#include "Compartment.hpp"

Mechanism::Mechanism(uint64_t source_id)
{
    this->source_id = source_id;
}

Mechanism::Mechanism()
{
    this->source_id = -1;
}

// double Mechanism::mechanism_fxn(const Compartment* pre_comp,
//                                 const Compartment* post_comp,
//                                 const double global_time,
//                                 const uint64_t curr_step)
// {
// 	printf("My source id is %lu\n", this->source_id);
// 	return 0.0;
// }

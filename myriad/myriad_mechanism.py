"""
Mechanism specification
"""

import logging

from .myriad_object import MyriadObject
from .myriad_metaclass import myriad_method_verbatim
from .myriad_types import MyriadScalar
from .myriad_types import MVoid, MVarArgs, MInt, MDouble

#######
# Log #
#######

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class Mechanism(MyriadObject):
    """
    Mechanism object specification.
    """
    source_id = MInt

    @myriad_method_verbatim
    def ctor(
            self,
            app: MyriadScalar("app", MVarArgs, ptr=True)
    ) -> MyriadScalar('', MVoid, ptr=True):
        """
    // Call superclass constructor first
    struct Mechanism* _self = (struct Mechanism*) super_ctor(MYRIADOBJECT, self, app);
    // Get source_id
    _self->source_id = va_arg(*app, uint_fast32_t);
    return self;
        """

    @myriad_method_verbatim
    def mechanism_calc(
            self,
            pre_comp: MyriadScalar("pre_comp", MVoid, True),
            global_time: MyriadScalar("gtime", MDouble, quals=["const"]),
            curr_step: MyriadScalar("cstep", MInt, quals=["const"])
    ) -> MDouble:
        """
        return 0.0;
        """

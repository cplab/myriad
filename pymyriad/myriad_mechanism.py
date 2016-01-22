"""
Mechanism specification
"""

import logging
import myriad_object

from myriad_metaclass import myriad_method_verbatim

from myriad_types import MyriadScalar
from myriad_types import MVoid, MVarArgs, MInt, MDouble

#######
# Log #
#######

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class Mechanism(myriad_object.MyriadObject):
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
        struct Mechanism* self =
            (struct Mechanism*) super_ctor(Mechanism, _self, app);
        self->source_id = va_arg(*app, uint64_t);
        return _self;
        """

    @myriad_method_verbatim
    def mechanism_calc(
            self,
            pre_comp: MyriadScalar("pre_comp", MVoid, True),
            post_comp: MyriadScalar("post_comp", MVoid, True),
            global_time: MyriadScalar("global_time", MDouble, quals=["const"]),
            curr_step: MyriadScalar("curr_step", MInt, quals=["const"])
    ) -> MDouble:
        """
        return 0.0;
        """


def main():
    """ Renders the Mechanism templates, logging the output to stderr """
    LOG.addHandler(logging.StreamHandler())
    Mechanism.render_templates()

if __name__ == "__main__":
    main()

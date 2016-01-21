"""
Compartment object specification
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


class Compartment(myriad_object.MyriadObject):
    """
    Compartment object specification.
    """
    cid = MInt

    @myriad_method_verbatim
    def ctor(
            self,
            app: MyriadScalar("app", MVarArgs, ptr=True)
    ) -> MyriadScalar('', MVoid, ptr=True):
        """
        struct Compartment* self =
            (struct Compartment*) super_ctor(Compartment, _self, app);
        self->cid = va_arg(*app, uint64_t);
        self->num_mechs = va_arg(*app, uint64_t);
        return self;
        """

    @myriad_method_verbatim
    def simul_fxn(
            self,
            network: MyriadScalar.void_ptr_ptr("network"),
            global_time: MyriadScalar("global_time", MDouble, quals=["const"]),
            curr_step: MyriadScalar("curr_step", MInt, quals=["const"])
    ) -> MDouble:
        """
        return 0.0;
        """

    @myriad_method_verbatim
    def add_mechanism(self, mechanism: MyriadScalar("mechanism", MVoid, True)):
        """
        if (_self == NULL || mechanism == NULL)
        {
            fputs("Neither Mechanism nor Compartment can be NULL.", stderr);
            return -1;
        }
        struct Compartment* self = (struct Compartment*) _self;
        struct Mechanism* mech = (struct Mechanism*) mechanism;

        if (self->num_mechs + 1 >= MAX_NUM_MECHS)
        {
            fputs("Cannot add mechanism to Compartment: out of room.", stderr);
            return -1;
        }
        self->my_mechs[self->num_mechs] = mech;
        self->num_mechs++;
        return 0;
        """


def main():
    """ Renders the Compartment templates, logging the output to stderr """
    LOG.addHandler(logging.StreamHandler())
    Compartment.render_templates()

if __name__ == "__main__":
    main()

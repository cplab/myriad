"""
Compartment object specification
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


class Compartment(MyriadObject):
    """
    Compartment object specification.
    """
    cid = MInt
    num_mechs = MInt
    my_mechs = MyriadScalar(ident="my_mechs",
                            base_type=MVoid,
                            ptr=True,
                            arr_id="MAX_NUM_MECHS")

    @myriad_method_verbatim
    def ctor(
            self,
            app: MyriadScalar("app", MVarArgs, ptr=True)
    ) -> MyriadScalar('', MVoid, ptr=True):
        """
        struct Compartment* _self =
            (struct Compartment*) super_ctor(Compartment, self, app);
        _self->cid = va_arg(*app, uint64_t);
        _self->num_mechs = va_arg(*app, uint64_t);
        return _self;
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
        if (self == NULL || mechanism == NULL)
        {
            fputs("Neither Mechanism nor Compartment can be NULL.", stderr);
            return;
        }
        struct Compartment* _self = (struct Compartment*) self;
        struct Mechanism* mech = (struct Mechanism*) mechanism;

        if (_self->num_mechs + 1 >= MAX_NUM_MECHS)
        {
            fputs("Cannot add mechanism to Compartment: out of room.", stderr);
            return;
        }
        _self->my_mechs[_self->num_mechs] = mech;
        _self->num_mechs++;
        """


def main():
    """ Renders the Compartment templates, logging the output to stderr """
    LOG.addHandler(logging.StreamHandler())
    Compartment.render_templates()

if __name__ == "__main__":
    main()

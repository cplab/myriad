"""
Compartment object specification
"""
import logging

from .myriad_object import MyriadObject
from .myriad_metaclass import myriad_method_verbatim
from .myriad_types import MyriadScalar
from .myriad_types import MVoid, MVarArgs, MInt, MDouble, MUInt

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
    num_mechs = MUInt
    mechs = MyriadScalar(ident="mechs",
                         base_type=MVoid,
                         ptr=True,
                         arr_id="MAX_NUM_MECHS")

    @myriad_method_verbatim
    def ctor(
            self,
            app: MyriadScalar("app", MVarArgs, ptr=True)
    ) -> MyriadScalar('', MVoid, ptr=True):
        """
    // Call superclass constructor first
    struct Compartment* _self = (struct Compartment*) super_ctor(MYRIADOBJECT, self, app);
    // Extract number of mechanisms
    _self->num_mechs = va_arg(*app, size_t);
    assert(_self->num_mechs <= MAX_NUM_MECHS);
    // Extract pointer to list of mechanisms and copy them
    void** mechs = va_arg(*app, void**);
    if (mechs)
    {
        memcpy(&_self->mechs, mechs, sizeof(void*) * _self->num_mechs);
    }
    return _self;
        """

    @myriad_method_verbatim
    def cudafy(self, cuda_self: MyriadScalar("cuda_self", MVoid, ptr=True)):
        """
#ifdef CUDA
    struct Compartment* _self = (struct Compartment*) self;

    // 1) Save our mechanism values in a temporary array
    MyriadObject_t tmp_mech_arr[_self->num_mechs];
    memset(&tmp_mech_arr, 0, _self->num_mechs * sizeof(struct MyriadObject*));
    memcpy(&tmp_mech_arr, &_self->mechs, _self->num_mechs * sizeof(void*));

    // 2) Allocate CUDA mechanism copies, overriding our object's pointers
    for (size_t i = 0; i < _self->num_mechs; i++)
    {
        _self->mechs[i] = myriad_cuda_new((struct MyriadObject*) _self->mechs[i]);
    }

    // 4) CUDAfy using MyriadObject, which will copy over the entire struct to
    //    the CUDA copy, including our device pointers
    super_myriad_cudafy(MYRIADOBJECT, _self, cuda_self);

    // 3) Copy back our mechanism pointers into our array
    memcpy(&_self->mechs, &tmp_mech_arr, sizeof(void*) * _self->num_mechs);
#else
    fputs("CUDAfication is not supported for non-CUDA targets.\\n", stderr);
#endif
        """

    @myriad_method_verbatim
    def decudafy(self, cuda_self: MyriadScalar("cuda_self", MVoid, ptr=True)):
        """
#ifdef CUDA
    // 0) We assume our child calls us first and copies their data AFTER
    struct Compartment* _self = (struct Compartment*) self;

    // 1) Make on-stack "blank" copy of the compartment header
    struct Compartment comp_cpy = {{MYRIADOBJECT}, 0, {NULL}};
    // 2) Copy over Compartment header of device object to our temp stack copy
    CUDA_CHECK_CALL(
        cudaMemcpy(
            &comp_cpy,
            cuda_self,
            sizeof(struct Compartment),
            cudaMemcpyDeviceToHost));
    // 3) DeCUDAfy using 'copy destructor', using our valid host objects as the
    //    host object as the location for the device to copy to
    for (size_t i = 0; i < _self->num_mechs; i++)
    {
        myriad_cuda_delete((struct MyriadObject*) _self->mechs[i], (struct MyriadObject*) comp_cpy.mechs[i]);
    }
    // 4) Remainder of comp_cpy is uninteresting to us so we don't copy it
#else
    fputs("deCUDAfication is not supported for non-CUDA targets.\\n", stderr);
#endif
        """

    @myriad_method_verbatim
    def dtor(self) -> MInt:
        """
    struct Compartment* _self = (struct Compartment*) self;
    for (size_t i = 0; i < _self->num_mechs; i++)
    {
        dtor(_self->mechs[i]);
    }
    return super_dtor(MYRIADOBJECT, self);
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


def main():
    """ Renders the Compartment templates, logging the output to stderr """
    LOG.addHandler(logging.StreamHandler())
    Compartment.render_templates()

if __name__ == "__main__":
    main()

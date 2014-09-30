"""
TODO: Docstring
"""

from collections import OrderedDict

import myriad_module

from myriad_types import MyriadScalar, MVoid, MUInt, MDouble, MyriadFunction

from MyriadObject import MyriadObject


class _Mechanism(myriad_module.MyriadModule):

    def __init__(self):
        # Object variables
        obj_vars = OrderedDict()
        obj_vars["source_id"] = MyriadScalar("source_id", MUInt)

        # Mechanism Function
        mech_fun_def = """
        const struct Mechanism* self = (const struct Mechanism*) _self;
        printf("My source id is %u\n", self->source_id);
        return 0.0;
        """
        mech_fun_args = OrderedDict()
        mech_fun_args["pre_comp"] = MyriadScalar("pre_comp",
                                                 MVoid,
                                                 ptr=True)
        mech_fun_args["post_comp"] = MyriadScalar("post_comp",
                                                  MVoid,
                                                  ptr=True)
        mech_fun_args["dt"] = MyriadScalar("dt",
                                           MDouble,
                                           quals=["const"])
        mech_fun_args["global_time"] = MyriadScalar("global_time",
                                                    MDouble,
                                                    quals=["const"])
        mech_fun_args["curr_step"] = MyriadScalar("curr_step",
                                                  MUInt,
                                                  quals=["const"])
        mech_fun_ret_var = MyriadScalar("", MDouble)

        # Methods
        methods = OrderedDict()
        methods["mechanism_fxn"] = MyriadFunction("mechanism_fxn",
                                                  mech_fun_args,
                                                  mech_fun_ret_var,
                                                  fun_def=mech_fun_def)
        super().__init__(MyriadObject,
                         "Mechanism",
                         obj_vars=obj_vars,
                         methods=methods)


Mechanism = _Mechanism()

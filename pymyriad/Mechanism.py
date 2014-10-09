"""
TODO: Docstring
"""

from collections import OrderedDict

import myriad_module

from myriad_types import MyriadScalar, MVoid, MUInt, MDouble, MyriadFunction

from MyriadObject import MyriadObject


class _Mechanism(myriad_module.MyriadModule):

    # Mechanism Function
    MECH_FUN_DEF = """
    const struct Mechanism* self = (const struct Mechanism*) _self;
    printf("My source id is %u\\n", self->source_id);
    return 0.0;
    """

    CTOR_FUN_DEF = """
    struct Mechanism* self = (struct Mechanism*) super_ctor(Mechanism, _self, app);
    self->source_id = va_arg(*app, unsigned int);
    return _self;
    """

    def __init__(self):
        # Object variables
        obj_vars = OrderedDict()
        obj_vars["source_id"] = MyriadScalar("source_id", MUInt)

        # Ctor, overrides instance method only
        _ctor_args = MyriadObject.methods["myriad_ctor"].delegator.args_list
        _ctor_args["source_id"] = MyriadScalar("source_id", MUInt)
        _ctor_ret_var = MyriadObject.methods["myriad_ctor"].delegator.ret_var
        mech_ctor = MyriadFunction("myriad_ctor",
                                   _ctor_args,
                                   _ctor_ret_var,
                                   fun_def=_Mechanism.CTOR_FUN_DEF)

        # Mechanism function (new function)
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
        methods["myriad_ctor"] = mech_ctor
        methods["mechanism_fxn"] = MyriadFunction("mechanism_fxn",
                                                  mech_fun_args,
                                                  mech_fun_ret_var,
                                                  fun_def=self.MECH_FUN_DEF)
        super().__init__(MyriadObject,
                         "Mechanism",
                         obj_vars=obj_vars,
                         methods=methods)


Mechanism = _Mechanism()


def main():
    # TESTING
    Mechanism.header_template.render_to_file()
    Mechanism.c_file_template.render_to_file()

if __name__ == "__main__":
    main()

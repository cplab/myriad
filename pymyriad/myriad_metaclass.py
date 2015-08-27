"""
.. module:: myriad_metaclass
    :platform: Linux
    :synposis: Provides metaclass for automatic Myriad integration

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>


"""

from copy import copy

from myriad_mako_wrapper import MakoTemplate

from myriad_types import MyriadScalar, MyriadFunction
from myriad_types import MVoid


DELG_TEMPLATE = open("templates/delegator_func.mako", 'r').read()

SUPER_DELG_TEMPLATE = open("templates/super_delegator_func.mako", 'r').read()


def create_super_delegator(m_fxn: MyriadFunction):
    """
    Create super delegator function.
    """
    super_args = copy(m_fxn.args_list)
    super_class_arg = MyriadScalar("_class", MVoid, True, ["const"])
    tmp_arg_indx = len(super_args) + 1
    super_args[tmp_arg_indx] = super_class_arg
    super_args.move_to_end(tmp_arg_indx, last=False)

    # Generate template and render
    template_vars = {}
    template = MakoTemplate(SUPER_DELG_TEMPLATE, template_vars)
    template.render()
    return MyriadFunction("super_" + m_fxn.ident,
                          super_args,
                          m_fxn.ret_var,
                          fun_def=template.buffer)


def gen_instance_method_from_str(delegator, m_name: str, method_body: str):
    """
    Automatically generate a MyriadFunction wrapper for a method body.

    :param str m_name: Name to prepend to the instance method identifier.
    :param str method_body: String template to use as the method body.
    """
    return MyriadFunction(m_name + '_' + delegator.ident,
                          args_list=delegator.args_list,
                          ret_var=delegator.ret_var,
                          storage=['static'],
                          fun_def=method_body)

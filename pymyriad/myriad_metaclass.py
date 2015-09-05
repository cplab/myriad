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


def create_delegator(instance_fxn: MyriadFunction, classname: str):
    """
    Creates a delegator function based on a function definition.
    """
    # Create copy with modified identifier
    ist_cpy = MyriadFunction.from_myriad_func(
        instance_fxn,
        ident=instance_fxn.ident.partition(classname + "_")[-1])
    # Generate template and render into copy's definition
    template_vars = {"delegator": ist_cpy, "classname": classname}
    template = MakoTemplate(DELG_TEMPLATE, template_vars)
    template.render()
    ist_cpy.fun_def = template.buffer
    # Return created copy
    return ist_cpy


def create_super_delegator(m_fxn: MyriadFunction, classname: str):
    """
    Create super delegator function.
    """
    super_args = copy(m_fxn.args_list)
    super_class_arg = MyriadScalar("_class", MVoid, True, ["const"])
    tmp_arg_indx = len(super_args) + 1
    super_args[tmp_arg_indx] = super_class_arg
    super_args.move_to_end(tmp_arg_indx, last=False)

    # Generate template and render
    delegator_f = MyriadFunction("super_" + m_fxn.ident,
                                 super_args,
                                 m_fxn.ret_var)
    template_vars = {"super_delegator": delegator_f, "classname": classname}
    template = MakoTemplate(SUPER_DELG_TEMPLATE, template_vars)
    template.render()

    # Add rendered definition to function
    delegator_f.fun_def = template.buffer
    return delegator_f


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

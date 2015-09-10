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

    :param 
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


def create_super_delegator(delg_fxn: MyriadFunction, classname: str):
    """
    Create super delegator function.

    :param MyriadFunction delg_fxn: Delegator to create super_* wrapper for
    :param str classname: Name of the base class for this super delegator

    :return: Super delegator method as a MyriadFunction
    :rtype: MyriadFunction
    """
    # Create copy of delegator function with modified parameters
    super_args = copy(delg_fxn.args_list)
    super_class_arg = MyriadScalar("_class", MVoid, True, ["const"])
    tmp_arg_indx = len(super_args) + 1
    super_args[tmp_arg_indx] = super_class_arg
    super_args.move_to_end(tmp_arg_indx, last=False)
    s_delg_f = MyriadFunction.from_myriad_func(delg_fxn,
                                               "super_" + delg_fxn.ident,
                                               super_args)

    # Generate template and render
    template_vars = {"delegator": delg_fxn,
                     "super_delegator": s_delg_f,
                     "classname": classname}
    template = MakoTemplate(SUPER_DELG_TEMPLATE, template_vars)
    template.render()

    # Add rendered definition to function
    s_delg_f.fun_def = template.buffer
    return s_delg_f


def gen_instance_method_from_str(delegator, m_name: str, method_body: str):
    """
    Automatically generate a MyriadFunction wrapper for a method body.

    :param str m_name: Name to prepend to the instance method identifier
    :param str method_body: String template to use as the method body

    :return: Instance method as a MyriadFunction
    :rtype: MyriadFunction
    """
    return MyriadFunction(m_name + '_' + delegator.ident,
                          args_list=delegator.args_list,
                          ret_var=delegator.ret_var,
                          storage=['static'],
                          fun_def=method_body)

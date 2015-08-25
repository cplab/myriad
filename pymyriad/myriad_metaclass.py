"""
.. module:: myriad_metaclass
    :platform: Linux
    :synposis: Provides metaclass for automatic Myriad integration

.. moduleauthor:: Pedro Rittner <pr273@cornell.edu>


"""

import copy

from myriad_utils import enforce_annotations
from myriad_mako_wrapper import MakoTemplate

from myriad_types import MyriadScalar, MyriadFunction
from myriad_types import MVoid


# pylint: disable=R0903
class MyriadMethod(object):
    """
    Generic class for abstracting methods into 3 core components:

    1. Delegator - API entry point for public method calls
    2. Super Delegator - API entry point for subclasses
    3. Instance Function(s) - 'Internal' method definitions for each class

    It also contains internal templates for generating delegators and super
    delegators from scratch.
    """

    DELG_TEMPLATE = open("templates/delegator_func.mako", 'r').read()

    SUPER_DELG_TEMPLATE = \
        open("templates/super_delegator_func.mako", 'r').read()

    @enforce_annotations
    def __init__(self,
                 m_fxn: MyriadFunction,
                 instance_methods: dict=None,
                 inherited: bool=False):
        """
        Initializes a method from a function.

        The point of this class is to automatically create delegators for a
        method. This makes inheritance of methods easier since the delegators
        are not implemented by the subclass, only the instance methods are
        overwritten.

        Note that inherited methods do not create delegators, only instance
        methods.

        :param MyriadFunction m_fxn: Method's prototypical delegator function.
        :param dict instance_methods: Mapping of object/class names to methods.
        :param bool inherited: Flag for denoting this method is overloaded.
        """

        #: Flag is true if this method overrides a previously-defined method.
        self.inherited = inherited

        # Need to ensure this function has a typedef
        m_fxn.gen_typedef()

        #: Local storage for delegator function.
        self.delegator = m_fxn

        #: Mapping of object/class names to instance methods for each.
        self.instance_methods = {}

        # Initialize (default: None) instance method(s)
        for obj_name, i_method in instance_methods.items():
            # If we are given a string, assume this is the instance method body
            # and auto-generate the MyriadFunction wrapper.
            if isinstance(i_method, str):
                self.gen_instance_method_from_str(obj_name, i_method)
            else:
                raise NotImplementedError("Non-string instance methods.")

        # Create super delegator
        super_args = copy.copy(m_fxn.args_list)
        super_class_arg = MyriadScalar("_class", MVoid, True, ["const"])
        tmp_arg_indx = len(super_args) + 1
        super_args[tmp_arg_indx] = super_class_arg
        super_args.move_to_end(tmp_arg_indx, last=False)
        _delg = MyriadFunction("super_" + m_fxn.ident,
                               super_args,
                               m_fxn.ret_var)
        #: Automatically-generated super delegator for this method
        self.super_delegator = _delg
        #: Template for this method's delegator
        self.delg_template = MakoTemplate(self.DELG_TEMPLATE, vars(self))
        #: Template for this method's super delegator
        self.super_delg_template = MakoTemplate(self.SUPER_DELG_TEMPLATE,
                                                vars(self))

        # TODO: Implement instance method template
        #: Template for method's instance methods.
        self.instance_method_template = None

    def gen_instance_method_from_str(self,
                                     m_name: str,
                                     method_body: str):
        """
        Automatically generate a MyriadFunction wrapper for a method body.

        :param str m_name: Name to prepend to the instance method identifier.
        :param str method_body: String template to use as the method body.
        """
        _tmp_f = MyriadFunction(m_name + '_' + self.delegator.ident,
                                args_list=self.delegator.args_list,
                                ret_var=self.delegator.ret_var,
                                storage=['static'],
                                fun_def=method_body)
        self.instance_methods[m_name] = _tmp_f

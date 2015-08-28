<%doc>
    Expected values:
    delegator - MyriadFunction object representing the base delegator
    super_delegator - MyriadFunction object representing this function
    classname - Name of the class this is implemented for as a string
</%doc>
<%
    ## Get the function arguments as a comma-seperated list
    fun_args = ','.join([arg.ident for arg in super_delegator.args_list.values()][1:])
    ## Get the 'class' argument (usually '_class')
    class_arg = list(super_delegator.args_list.values())[0].ident
    ## Get the return type of this function
    ret_type = super_delegator.ret_var.base_type
%>
${super_delegator.stringify_decl()}
{
    const struct ${classname}* superclass = (const struct ${classname}*)
        myriad_super(${class_arg});

    ## Make sure that we return only for non-pointer void
    % if ret_type is MVoid and ret_type.ptr is False:
    superclass->my_${delegator.fun_typedef.name}(${fun_args});
    return;
    % else:
    return superclass->my_${delegator.fun_typedef.name}(${fun_args});
    % endif
}

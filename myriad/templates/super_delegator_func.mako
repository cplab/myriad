<%doc>
    Expected values:
    delegator - MyriadFunction object representing the base delegator
    super_delegator - MyriadFunction object representing this function
    classname - Name of the class this is implemented for as a string
    MVoid - myriad_types' MVoid type, passed by reference to avoid importing
</%doc>
<%
    ## Get the function arguments as a comma-seperated list
    fun_args = ','.join([arg.ident for arg in super_delegator.args_list.values()][1:])
    ## Get the 'class' argument (usually '_class')
    class_arg = list(super_delegator.args_list.values())[0].ident
    ## Get the return variable type of this function
    ret_var = super_delegator.ret_var
    ## Get the name of the vtable
    vtable_name = str(delegator.ident) + "_vtable"
%>
${super_delegator.stringify_decl()}
{
## Make sure that we return only for non-pointer void
% if ret_var.base_type is MVoid and not ret_var.ptr:
    ${vtable_name}[${class_arg}](${fun_args});
    return;
% else:
    return ${vtable_name}[${class_arg}](${fun_args});
% endif
}

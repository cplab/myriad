<%doc>
    Expected values:
    delegator - MyriadFunction object representing the base delegator
    classname - Name of the class this is implemented for as a string
    MVoid - myriad_types' MVoid type, passed by reference to avoid importing
</%doc>
<%
    ## Get the function arguments as a comma-seperated list
    fun_args = ','.join([arg.ident for arg in delegator.args_list.values()][1:])
    ## Get the 'class' argument (usually '_class')
    class_arg = list(delegator.args_list.values())[0].ident
    ## Get the return type of this function
    ret_var = delegator.ret_var
    ## Get the name of the vtable
    vtable_name = str(delegator.fun_typedef.name) + "_vtable"
%>
${delegator.stringify_decl()}
{
% if ret_var.base_type is MVoid and not ret_var.ptr:
    ${vtable_name}[((MyriadObject_t) obj)->class_id](self, ${fun_args});
    return;
% elif fun_args:
    return ${vtable_name}[((MyriadObject_t) obj)->class_id](self, ${fun_args});
% else:
    return ${vtable_name}[((MyriadObject_t) obj)->class_id](self);
% endif
}

<%doc>
    Expected values:
    delegator - MyriadFunction object representing the base delegator
    classname - Name of the class this is implemented for as a string
</%doc>
<%
    from context import myriad
    from myriad.myriad_types import MVoid
    ## Get the function arguments as a comma-seperated list
    fun_args = ','.join([arg.ident for arg in delegator.args_list.values()][1:])
    ## Get the 'class' argument (usually '_class')
    class_arg = list(delegator.args_list.values())[0].ident
    ## Get the return type of this function
    ret_var = delegator.ret_var
%>
${delegator.stringify_decl()}
{
    const struct ${classname}* m_class = (const struct ${classname}*)
        myriad_class_of(${class_arg});

    assert(m_class->my_${delegator.fun_typedef.name});

% if ret_var.base_type is MVoid and not ret_var.ptr:
    m_class->my_${delegator.fun_typedef.name}(self, ${fun_args});
    return;
% elif fun_args:
    return m_class->my_${delegator.fun_typedef.name}(self, ${fun_args});
% else:
    return m_class->my_${delegator.fun_typedef.name}(self);
% endif
}

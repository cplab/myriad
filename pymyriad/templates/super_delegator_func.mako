% if not inherited:
<%
    fun_args = ','.join([arg.ident for arg in super_delegator.args_list.values()][1:])
%>
${super_delegator.stringify_decl()}
{
    const struct MyriadClass* superclass = (const struct MyriadClass*)
        myriad_super(${list(super_delegator.args_list.values())[0].ident});

    assert(superclass->my_${delegator.fun_typedef.name});

    % if delegator.ret_var.base_type is MVoid and not delegator.ret_var.base_type.ptr:
    superclass->my_${delegator.fun_typedef.name}(${fun_args});
    return;
    % else:
    return superclass->my_${delegator.fun_typedef.name}(${fun_args});
    % endif
}
% endif
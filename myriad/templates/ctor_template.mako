    struct ${obj_name}* _self = (struct ${obj_name}*) super_ctor(${super_obj_name.upper()}, self, app);

% for var_name, var in myriad_obj_vars.items():
    % if var.arr_id is None and not var.ident.startswith('_'):
    _self->${var_name} = va_arg(*app, ${var.base_type.mtype.names[0]});
    % endif
% endfor

    return _self;

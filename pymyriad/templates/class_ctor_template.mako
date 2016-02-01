    struct ${cls_name}* _self = (struct ${cls_name}*) super_ctor(${cls_name}, self, app);

    voidf selector = NULL; selector = va_arg(*app, voidf);

    while (selector)
    {
        const voidf method = va_arg(*app, voidf);

% for mtd in own_methods:
        if (selector == (voidf) ${mtd.ident})
        {
            *(voidf *) &_self->${"my_" + mtd.typedef_name} = method;
        }
% endfor

        selector = va_arg(*app, voidf);
    }

    return _self;
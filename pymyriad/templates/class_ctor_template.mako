    struct ${cls_name}* self = (struct ${cls_name}*) super_ctor(${cls_name}, _self, app);

    voidf selector = NULL; selector = va_arg(*app, voidf);

    while (selector)
    {
        const voidf method = va_arg(*app, voidf);

% for mtd in own_methods:
        if (selector == (voidf) ${mtd.ident})
        {
            *(voidf *) &self->${"my_" + mtd.typedef_name} = method;
        }
% endfor

        selector = va_arg(*app, voidf);
    }

    return self;
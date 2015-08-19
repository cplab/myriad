struct ${cls_name}* self = (struct ${cls_name}*) super_ctor(${cls_name}, _self, app);

voidf selector = NULL; selector = va_arg(*app, voidf);

while (selector)
{
    const voidf method = va_arg(*app, voidf);

    % for mtd in [m for m in methods.values() if not m.inherited]:
    if (selector == (voidf) ${mtd.delegator.ident})
    {
        *(voidf *) &self->${"my_" + mtd.delegator.fun_typedef.name} = method;
    }
    % endfor

    selector = va_arg(*app, voidf);
}

return self;
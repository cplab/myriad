static struct ${cls_name} object[] =
{
    {
        { object + 1 },
        object,
        NULL,
        sizeof(struct ${obj_name}),
% for name in [obj_name + "_" + m.ident for m in myriad_methods.values() if not hasattr(m, "is_myriadclass_method")]:
        ${name},
% endfor
    },
    {
        { object + 1 },
        object,
        NULL,
        sizeof(struct ${cls_name}),
% for name in [obj_name + "_" + m.ident for m in myriad_methods.values() if hasattr(m, "is_myriadclass_method")]:
        ${name},
% endfor
    }
};


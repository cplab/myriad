#ifdef CUDA
    struct ${cls_name}* my_class = (struct ${cls_name}*) _self;

    struct ${cls_name} copy_class = *my_class; // Assignment to stack avoids calloc/memcpy
    struct MyriadClass* copy_class_class = (struct MyriadClass*) &copy_class;

% for method in own_methods:
    ${method.typedef_name} my_${method.ident}_fun = NULL;
    CUDA_CHECK_RETURN(
        cudaMemcpyFromSymbol(
            (void**) &my_${method.ident}_fun,
            (const void*) &${cls_name}_cuda_${method.ident},
            sizeof(void*),
            0,
            cudaMemcpyDeviceToHost
        )
    );
    copy_class.my_${method.typedef_name} = my_${method.ident}_fun;
% endfor

    if (clobber)
    {
        const struct MyriadClass* super_class = (const struct MyriadClass*) MyriadClass;
        memcpy((void**) &copy_class_class->super, &super_class->device_class, sizeof(void*));
    }
    return super_cudafy(${cls_name}, (void*) &copy_class, 0);

#else
    return NULL;
#endif
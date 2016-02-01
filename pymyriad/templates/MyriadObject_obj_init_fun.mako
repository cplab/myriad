#ifdef CUDA
    const struct MyriadClass *obj_addr = NULL, *class_addr = NULL;
    const size_t obj_size = sizeof(struct MyriadObject);
    const size_t class_size = sizeof(struct MyriadClass);

    CUDA_CHECK_RETURN(cudaMalloc((void**)&obj_addr, class_size));
    CUDA_CHECK_RETURN(cudaMalloc((void**)&class_addr, class_size));

    const struct MyriadClass anon_class_class = {
        {class_addr},
        obj_addr,
        class_addr,
        class_size,
        NULL,
        NULL,
        NULL,
        NULL,
    };

    CUDA_CHECK_RETURN(
        cudaMemcpy(
            (void**) class_addr,
            &anon_class_class,
            sizeof(struct MyriadClass),
            cudaMemcpyHostToDevice
            )
        );

    object[1].device_class = class_addr;

    const struct MyriadClass anon_obj_class = {
        {class_addr},
        obj_addr,
        class_addr,
        obj_size,
        NULL,
        NULL,
        NULL,
        NULL,
    };

    CUDA_CHECK_RETURN(
        cudaMemcpy(
            (void**) obj_addr,
            &anon_obj_class,
            sizeof(struct MyriadClass),
            cudaMemcpyHostToDevice
            )
        );

    object[0].device_class = (const struct MyriadClass*) obj_addr;

    CUDA_CHECK_RETURN(
        cudaMemcpyToSymbol(
            (const void*) &MyriadClass_dev_t,
            &class_addr,
            sizeof(void*),
            0,
            cudaMemcpyHostToDevice
            )
        );

    CUDA_CHECK_RETURN(
        cudaMemcpyToSymbol(
            (const void*) &MyriadObject_dev_t,
            &obj_addr,
            sizeof(void*),
            0,
            cudaMemcpyHostToDevice
            )
        );
#endif
    return;

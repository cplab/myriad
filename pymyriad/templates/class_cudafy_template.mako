#ifdef CUDA
{
    struct MechanismClass* my_class = (struct MechanismClass*) _self;

    struct MechanismClass copy_class = *my_class; // Assignment to stack avoids calloc/memcpy
    struct MyriadClass* copy_class_class = (struct MyriadClass*) &copy_class;

    mech_fun_t my_mech_fun = NULL;
    CUDA_CHECK_RETURN(
        cudaMemcpyFromSymbol(
            (void**) &my_mech_fun,
            (const void*) &Mechanism_cuda_mechanism_fxn_t,
            sizeof(void*),
            0,
            cudaMemcpyDeviceToHost
        )
    );
    copy_class.m_mech_fxn = my_mech_fun;

    if (clobber)
    {
        const struct MyriadClass* super_class = (const struct MyriadClass*) MyriadClass;
        memcpy((void**) &copy_class_class->super, &super_class->device_class, sizeof(void*));
    }
    return super_cudafy(MechanismClass, (void*) &copy_class, 0);
}
#else
{
    return NULL;
}
#endif
## Initializes the vtable using methods of children classes
% for method in own_methods:
void init_${method.ident}_cuvtable(void)
{
#ifdef CUDA
    mech_fun_t host_vtable[NUM_CU_CLASS] = { NULL };

    % for subclass in subclasses:
        % if method.ident in subclass.own_methods:
    CUDA_CHECK_CALL(cudaMemcpyFromSymbol(
                        &host_vtable[subclass.__name__.upper()],
                        Mechanism_mech_fun_devp,
                        sizeof(${method.typedef_name}),
                        0,
                        cudaMemcpyDeviceToHost));

    CUDA_CHECK_CALL(cudaMemcpyToSymbol(
                        ${method.ident}_vtable,
                        host_vtable,
                        sizeof(${method.typedef_name}) * NUM_CU_CLASS,
                        0,
                        cudaMemcpyHostToDevice
                        ));
        % endif
    % endfor
#endif
}
% endfor

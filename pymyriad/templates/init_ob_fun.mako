<%doc>
    Expected values:
    obj_name: object name
    cls_name: class name
    super_obj_name: superobject name
    super_cls_name: superclass name
</%doc>

void init${obj_name}(void)
{
	if (!${cls_name})
	{
		${cls_name} = 
			myriad_new(
				   ${super_cls_name},
				   ${super_cls_name},
				   sizeof(struct ${cls_name}),
                   ## TODO: Overwrite methods here
				   0
			);
		struct MyriadObject* class_obj = (struct MyriadObject*) ${cls_name};
		memcpy((void**) &class_obj->m_class, &${cls_name}, sizeof(void*));

#ifdef CUDA
	   	void* tmp_c = myriad_cudafy((void*)${cls_name}, 1);
		((struct MyriadClass*) CompartmentClass)->device_class = (struct MyriadClass*) tmp_c;
		CUDA_CHECK_RETURN(
			cudaMemcpyToSymbol(
				(const void*) &${cls_name}_dev_t,
				&tmp_c,
				sizeof(struct ${cls_name}*),
				0,
				cudaMemcpyHostToDevice
				)
			);
#endif
	}
	
	if (!${obj_name})
	{
		${obj_name} = 
			myriad_new(
				   ${cls_name},
				   ${super_obj_name},
				   sizeof(struct ${obj_name}),
                   ## TODO: Overwrite methods here
                   ## TODO: Add new functions here (own_methods) as 'delg_name, static_func'
				   0
			);
#ifdef CUDA
		void* tmp_o = myriad_cudafy((void*)${obj_name}, 1);
		((struct MyriadClass*) ${obj_name})->device_class = (struct MyriadClass*) tmp_o;
		CUDA_CHECK_RETURN(
			cudaMemcpyToSymbol(
				(const void*) &${obj_name}_dev_t,
				&tmp_o,
				sizeof(struct ${obj_name}*),
				0,
				cudaMemcpyHostToDevice
				)
			);
#endif
	}
}
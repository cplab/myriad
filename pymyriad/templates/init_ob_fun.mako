void init${obj_name}(void)
{
	if (!${cls_name})
	{
		${cls_name} = 
			myriad_new(
				   ${super_cls},
				   ${super_cls},
				   sizeof(struct ${cls_name}),
## Overwrite methods here
## FIXME: Get rid of this dirty hack for cls_* methods to work
% for method in myriad_methods.values():
    % if method.ident.startswith("cls_"):
                   ${method.ident[4:]}, ${obj_name}_${method.ident},
    % endif
% endfor
				   0
			);
		struct MyriadObject* class_obj = (struct MyriadObject*) ${cls_name};
		memcpy((void**) &class_obj->mclass, &${cls_name}, sizeof(void*));

#ifdef CUDA
	   	void* tmp_c = myriad_cudafy((void*)${cls_name}, 1);
		((struct MyriadClass*) ${cls_name})->device_class = (struct MyriadClass*) tmp_c;
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
				   ${super_obj},
				   sizeof(struct ${obj_name}),
## FIXME: Get rid of this dirty hack for cls_* methods to work
% for method in myriad_methods.values():
      % if not method.ident.startswith("cls_"):
                   ${method.ident}, ${obj_name}_${method.ident},
      % endif
% endfor
## Add new functions here (own_methods)
% for method in own_methods:
                   ${method.ident}, ${obj_name}_${method.ident},
% endfor
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
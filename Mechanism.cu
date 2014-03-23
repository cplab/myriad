#include <assert.h>
#include <string.h>
#include "Mechanism.cuh"

#include <cuda_runtime.h>

static void* Mechanism_ctor(void* _self, va_list* app)
{
	struct Mechanism* self = (struct Mechanism*) super_ctor(Mechanism, _self, app);
	
	self->source_id = va_arg(*app, unsigned int);

	return self;
}

static void* Mechanism_cudafy(void* _self, int clobber)
{
	//TODO: What value of clobber for non-class objects?
	return super_cudafy(Mechanism, _self, clobber); 
}

static double Mechanism_mechanism_fxn(
	void* _self,
    void* pre_comp,
    void* post_comp,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct Mechanism* self = (const struct Mechanism*) _self;
	printf("My source id is %u\n", self->source_id);
	return 0.0;
}

__device__ double Mechanism_cuda_mechanism_fxn(
	void* _self,
    void* pre_comp,
    void* post_comp,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct Mechanism* self = (const struct Mechanism*) _self;
	printf("My source id is %u\n", self->source_id);
	return 0.0;
}

// TODO: Make this extern? No clue.
__device__ mech_fun_t Mechanism_cuda_mechanism_fxn_t = Mechanism_cuda_mechanism_fxn;

double mechanism_fxn(
	void* _self,
    void* pre_comp,
    void* post_comp,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct MechanismClass* m_class = (const struct MechanismClass*) myriad_class_of(_self);
	assert(m_class->m_mech_fxn);
	return m_class->m_mech_fxn(_self, pre_comp, post_comp, dt, global_time, curr_step);
}

__device__ double cuda_mechanism_fxn(
	void* _self,
    void* pre_comp,
    void* post_comp,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct MechanismClass* m_class=(const struct MechanismClass*) cuda_myriad_class_of(_self);

	return m_class->m_mech_fxn(_self, pre_comp, post_comp, dt, global_time, curr_step);
}

double super_mechanism_fxn(
	void* _class,
	void* _self,
    void* pre_comp,
    void* post_comp,
    const double dt,
    const double global_time,
	const unsigned int curr_step
	)
{
	const struct MechanismClass* s_class=(const struct MechanismClass*) myriad_super(_class);
	assert(_self && s_class->m_mech_fxn);
	return s_class->m_mech_fxn(_self, pre_comp, post_comp, dt, global_time, curr_step);
}

static void* MechanismClass_ctor(void* _self, va_list* app)
{
	struct MechanismClass* self = (struct MechanismClass*) super_ctor(MechanismClass, _self, app);

	voidf selector;

	while ((selector = va_arg(*app, voidf)))
	{
		voidf method = va_arg(*app, voidf);
		
		if (selector == (voidf) mechanism_fxn)
		{
			*(voidf *) &self->m_mech_fxn = method;
		}
	}

	return self;
}

static void* MechanismClass_cudafy(void* _self, int clobber)
{
	void* result = NULL;
	
    // We know that we're actually a mechanism class
	struct MechanismClass* my_class = (struct MechanismClass*) _self;

	// Make a temporary copy-class because we need to change shit
	struct MechanismClass* copy_class = (struct MechanismClass*) calloc(1, sizeof(struct MechanismClass));
	memcpy((void**) copy_class, my_class, sizeof(struct MechanismClass));
	struct MyriadClass* copy_class_class = (struct MyriadClass*) copy_class;

	// TODO: Find a better way to get function pointers for on-card functions
	mech_fun_t my_mech_fun = NULL;
	CUDA_CHECK_RETURN(
		cudaMemcpyFromSymbol(
			(void**) &my_mech_fun,
			Mechanism_cuda_mechanism_fxn_t,
			sizeof(void*)
			)
		);
	copy_class->m_mech_fxn = my_mech_fun;
	printf("Copy Class mech fxn: %p\n", my_mech_fun);
	
	// !!!!!!!!! IMPORTANT !!!!!!!!!!!!!!
	// By default we clobber the copy_class_class' superclass with
	// the superclass' device_class' on-GPU address value. 
	// To avoid cloberring this value (e.g. if an underclass has already
	// clobbered it), the clobber flag should be 0.
	if (clobber)
	{
		const struct MyriadClass* super_class = (const struct MyriadClass*) MyriadClass;
		memcpy((void**) &copy_class_class->super, &super_class->device_class, sizeof(void*));
	}

	// This works because super methods rely on the given class'
	// semi-static superclass definition, not it's ->super attribute.
	result = super_cudafy(MechanismClass, (void*) copy_class, 0);

	free(copy_class); // No longer needed, stuff is on the card now
	
	return result;
}

const void *MechanismClass, *Mechanism;

__device__ __constant__ struct Mechanism* Mechanism_dev_t = NULL;
__device__ __constant__ struct MechanismClass* MechanismClass_dev_t = NULL;

void initMechanism(int init_cuda)
{
	if (!MechanismClass)
	{
		MechanismClass = 
			myriad_new(
				   MyriadClass,
				   MyriadClass,
				   size_t(sizeof(struct MechanismClass)),
				   myriad_ctor, MechanismClass_ctor,
				   myriad_cudafy, MechanismClass_cudafy,
				   0
			);
		struct MyriadObject* mech_class_obj = (struct MyriadObject*) MechanismClass;
		memcpy( (void**) &mech_class_obj->m_class, &MechanismClass, sizeof(void*));

		// TODO: Additional checks for CUDA initialization
		if (init_cuda)
		{
			void* tmp_mech_c_t = myriad_cudafy((void*)MechanismClass, 1);
			((struct MyriadClass*) MechanismClass)->device_class = (struct MyriadClass*) tmp_mech_c_t;
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					Mechanism_dev_t,
					&tmp_mech_c_t,
					sizeof(struct MechanismClass*),
					size_t(0),
					cudaMemcpyHostToDevice
					)
				);
		}
	}
	
	if (!Mechanism)
	{
		Mechanism = 
			myriad_new(
				   MechanismClass,
				   MyriadObject,
				   sizeof(struct Mechanism),
				   myriad_ctor, Mechanism_ctor,
				   myriad_cudafy, Mechanism_cudafy,
				   mechanism_fxn, Mechanism_mechanism_fxn,
				   0
			);

		if (init_cuda)
		{
			void* tmp_mech_t = myriad_cudafy((void*)Mechanism, 1);
			((struct MyriadClass*) Mechanism)->device_class = (struct MyriadClass*) tmp_mech_t;
			CUDA_CHECK_RETURN(
				cudaMemcpyToSymbol(
					Mechanism_dev_t,
					&tmp_mech_t,
					sizeof(struct Mechanism*),
					size_t(0),
					cudaMemcpyHostToDevice
					)
				);

		}
	}
	
}

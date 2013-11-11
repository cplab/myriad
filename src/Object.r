#ifndef OBJECT_R
#define OBJECT_R

#include "Object.h"

struct Object
{
	const struct Class* class;	/* object's description */
};

struct Class
{
	const struct Object _;			/* class' description */
	const char* name;				/* class' name */
	const struct Class* super;		/* class' super class */
	size_t size;					/* class' object's size */
	void* (* ctor) (void* self, va_list* app);
	void* (* dtor) (void* self);
	int (* differ) (const void* self, const void* b);
	int (* puto) (const void* self, FILE* fp);
};

extern void* super_ctor (const void* class, void* self, va_list* app);
extern void* super_dtor (const void* class, void* self);
extern int super_differ (const void* class, const void* self, const void* b);
extern int super_puto (const void* class, const void* self, FILE* fp);

#endif

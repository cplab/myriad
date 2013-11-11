#ifndef	OBJECT_H
#define	OBJECT_H

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef void (* voidf) ();	/* generic function pointer */

extern const void * Object;		/* new(Object); */

extern void* new (const void* class, ...);
extern void delete (void* self);

extern const void* classOf (const void* self);
extern size_t sizeOf (const void* self);

extern int isA (const void* _self, const void* class);	// object belongs to class
extern int isOf (const void*  _self, const void* class);// object derives from class

extern void* ctor (void* self, va_list * app);
extern void* dtor (void* self);
extern int differ (const void* self, const void* b);
extern int puto (const void* self, FILE* fp);

extern const void * Class;	/* new(Class, "name", super, size, sel, meth, ... 0); */

extern const void* super (const void* self);	/* class' superclass */

#endif

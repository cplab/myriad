## Add include guards
<% include_guard = obj_name.upper() + "_H" %>
#ifndef ${include_guard}
#define ${include_guard}

## Top-level Myriad include
#include "myriad.h"

## Add library includes
% for lib in lib_includes:
#include <${lib}>
% endfor

## Add local includes
% for lib in local_includes:
#include "${lib}"
% endfor

## Add parent class include
% if super_obj_name:
#include "${super_obj_name}.cuh"
% endif

## Class/Object structs, with special case for MyriadObject
% if obj_name != "MyriadObject":
${obj_struct.stringify_decl()};
% else:
struct MyriadObject
{
    const enum MyriadClass class_id;
};
% endif

## Declare typedefs/vtables/init functions for own methods ONLY
% for (delg, _) in own_method_delgs:

#ifndef ${delg.typedef_name.upper()}
#define ${delg.typedef_name.upper()}
${delg.stringify_typedef()};
#endif

extern
% if obj_name != "MyriadObject":
#ifdef CUDA
__constant__
#endif
% endif
const ${delg.typedef_name} ${delg.ident}_vtable[NUM_CU_CLASS];

## Top-level init functions for vtable
extern void init_${delg.ident}_cuvtable(void);

% endfor

## Process instance methods
<%
instance_methods = [m.from_myriad_func(m, obj_name + "_" + m.ident) for m in myriad_methods.values()]
%>

## Print methods forward declarations
% for mtd in instance_methods:
extern ${mtd.stringify_decl()};
% endfor

## Method delegators
% for delg, super_delg in own_method_delgs:
extern ${delg.stringify_decl()};

extern ${super_delg.stringify_decl()};

% endfor

#endif

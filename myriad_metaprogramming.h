/**
   @author Pedro Rittner, Alex J Davies
 */

#define _MYRIAD_CAT_HELPER(x,y) x ## y
#define _MYRIAD_CAT(x, y) _MYRIAD_CAT_HELPER(x,y)

#define MYRIAD_FXN_TYPEDEF_GEN(name, args, ret) \
    ret (* name)(args)

#define MYRIAD_FXN_METHOD_HEADER_GEN(ret, args, objname, suffix) \
    ret _MYRIAD_CAT(objname, _MYRIAD_CAT(_,suffix)) (args)

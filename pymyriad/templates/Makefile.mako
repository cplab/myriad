################################
##     CUDA Binaries & Libs   ##
################################
CUDA_PATH := ${CUDA_PATH}
CUDA_INC_PATH	?= $(CUDA_PATH)/include
CUDA_BIN_PATH	?= $(CUDA_PATH)/bin
CUDA_LIB_PATH	?= $(CUDA_PATH)/lib64

################################
##      Compilers & Tools     ##
################################
NVCC	?= $(CUDA_BIN_PATH)/nvcc
CC	    := ${CC}
CXX	    := ${CXX}
AR	    ?= ar

################################
##       COMPILER FLAGS       ##
################################

## OS Arch Flags
OS_SIZE = ${OS_SIZE}
OS_ARCH = ${OS_ARCH}

## CC & related flags, with debug switch
COMMON_CFLAGS := -Wall -Wextra -Wno-unused-parameter -Wno-format -Wno-unknown-pragmas

% if DEBUG:
COMMON_CFLAGS += -O0 -g3 -DDEBUG=1$(DEBUG)
% else:
COMMON_CFLAGS += -DNDEBUG -O2 -march=native
% endif

## Link-time optimization
COMMON_CFLAGS += -flto

## To Consider:
## -ffinite-math-only : Assume no -Inf/+Inf/NaN
## -fno-trapping-math : Floating-point operations cannot generate OS traps
## -fno-math-errno : Don't set errno for <math.h> functions
CCFLAGS		:= $(COMMON_CFLAGS) -Wpedantic -std=gnu99
CXXFLAGS	:= $(COMMON_CFLAGS) -std=c++11
CUFLAGS		:= $(COMMON_CFLAGS)

## NVCC & related flags
NVCC_HOSTCC_FLAGS = -x cu -ccbin $(CC) $(addprefix -Xcompiler , $(CUFLAGS))
NVCCFLAGS := -m$(OS_SIZE)

% if DEBUG:
NVCCFLAGS += -g -G
% endif

## TODO: Make sure CUDA compute architecture flags are set correctly
GENCODE_FLAGS := -gencode arch=compute_30,code=sm_30
EXTRA_NVCC_FLAGS := -rdc=true

## CPU Myriad Objects
MYRIAD_OBJS 	:= myriad_alloc.o mmq.o ${myriad_lib_objs}

################################
##      Linker (LD) Flags     ##
################################

LD_FLAGS            := -L. -lm -lpthread -lrt
CUDART_LD_FLAGS     := -L$(CUDA_LIB_PATH) -lcudart
CUDA_BIN_LDFLAGS    := $(CUDART_LD_FLAGS) $(LD_FLAGS)

################################
##       Definition Flags     ##
################################

DEFINES ?=
% if CUDA:
DEFINES += -DCUDA
% endif

CUDA_DEFINES := $(DEFINES)
CUDA_BIN_DEFINES ?= $(DEFINES)

################################
##    Include Path & Flags    ##
################################

CUDA_INCLUDES := -I$(CUDA_INC_PATH)
INCLUDES := $(CUDA_INCLUDES) -I.

################################
##        Make Targets        ##
################################

SIMUL_MAIN := main
SIMUL_MAIN_OBJ := $(addsuffix .o, $(SIMUL_MAIN))
SIMUL_MAIN_BIN := $(addsuffix .bin, $(SIMUL_MAIN))

CUDA_LINK_OBJ := 
% if CUDA:
CUDA_LINK_OBJ	+= dlink.o
% endif

OBJECTS		:= $(wildcard *.o)
LIBRARIES	:= $(wildcard *.a)
BINARIES	:= $(wildcard *.bin)

################################
##         Make Rules         ##
################################

## ------- Build Rules -------

.PHONY: clean all build

build: all

all: $(SIMUL_MAIN_BIN)

clean:
	@rm -f $(OBJECTS) $(LIBRARIES) $(BINARIES) *.s *.i *.ii

## ------- CPU Myriad Objects -------

$(MYRIAD_OBJS): %.o : %.c
	$(CC) $(CCFLAGS) $(INCLUDES) $(DEFINES) -o $@ -c $<

## ------- CUDA Myriad Library -------

$(CUDA_MYRIAD_LIB): $(CUDA_MYRIAD_LIB_OBJS)
	$(NVCC) -lib $^ -o $(CUDA_MYRIAD_LIB)

$(CUDA_MYRIAD_LIB_OBJS): %.o : %.cu
	$(NVCC) $(NVCC_HOSTCC_FLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) \
	$(CUDA_INCLUDES) $(CUDA_DEFINES) -o $@ -dc $<

## ------- Linker Object -------

$(CUDA_LINK_OBJ): $(SIMUL_MAIN_OBJ) $(CUDA_MYRIAD_LIB_OBJS)
	$(NVCC) $(GENCODE_FLAGS) -dlink $^ -o $(CUDA_LINK_OBJ) # Needed for seperate compilation

## ------- Main binary object -------

$(SIMUL_MAIN_OBJ): %.o : %.c
% if CUDA:
	$(NVCC) $(NVCC_HOSTCC_FLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) \
	$(CUDA_INCLUDES) $(CUDA_BIN_DEFINES) -o $@ -dc $<
% else:
	$(CC) $(CCFLAGS) $(INCLUDES) $(DEFINES) -x c -c $< -o $@
% endif

## ------- Host Linker Generated Binary -------

$(SIMUL_MAIN_BIN): $(SIMUL_MAIN_OBJ) $(CUDA_LINK_OBJ) $(MYRIAD_OBJS) $(CUDA_MYRIAD_LIB)
% if CUDA:
	$(CC) -o $@ $+ $(CUDA_BIN_LDFLAGS)
% else:
    % if DEBUG:
	$(CC) -Og -g -flto -o $@ $+ $(LD_FLAGS)
    % else:
    $(CC) -O2 -flto -o $@ $+ $(LD_FLAGS)
    % endif
% endif


###############################
#     CUDA Binaries & Libs    #
###############################
CUDA_PATH	?= /usr/local/cuda
CUDA_INC_PATH	?= $(CUDA_PATH)/include
CUDA_BIN_PATH	?= $(CUDA_PATH)/bin
CUDA_LIB_PATH	?= $(CUDA_PATH)/lib64

###############################
#      Compilers & Tools      #
###############################
NVCC	?= $(CUDA_BIN_PATH)/nvcc
CC	:= x86_64-linux-gnu-gcc

###############################
#       COMPILER FLAGS        #
###############################

# OS Arch Flags
OS_SIZE = 64
OS_ARCH = x86_64

# CC & related flags, with debug switch
COMMON_CFLAGS := -Wall -Wextra -Wno-unused-parameter -Wno-unknown-pragmas

ifdef DEBUG
COMMON_CFLAGS += -Og -g -DDEBUG=$(DEBUG) -DUNIT_TEST
else
COMMON_CFLAGS += -O2 -march=native
endif


CCFLAGS		:= $(COMMON_CFLAGS) -Wpedantic -std=gnu99 -ffinite-math-only -fno-trapping-math -fno-math-errno
ifndef CUDA
CCFLAGS += -flto
endif
CUFLAGS		:= $(COMMON_CFLAGS)

# NVCC & related flags
NVCC_HOSTCC_FLAGS = --device-c -x cu -ccbin $(CC) $(addprefix -Xcompiler , $(CUFLAGS))
NVCCFLAGS := -m$(OS_SIZE)
ifdef DEBUG
NVCCFLAGS += -g -G
endif
GENCODE_FLAGS := -gencode arch=compute_35,code=sm_35
EXTRA_NVCC_FLAGS := -rdc=true

###############################
#        Libraries            #
###############################

# CPU Myriad Library
MYRIAD_LIB_OBJS	:= MyriadObject.c.o Mechanism.c.o Compartment.c.o \
	HHSomaCompartment.c.o HHLeakMechanism.c.o HHNaCurrMechanism.c.o HHKCurrMechanism.c.o \
	DCCurrentMech.c.o HHGradedGABAAMechanism.c.o HHSpikeGABAAMechanism.c.o myriad_alloc.c.o

# CUDA Myriad Library
CUDA_MYRIAD_LIB_OBJS = MyriadObject.cu.o Mechanism.cu.o Compartment.cu.o \
	HHSomaCompartment.cu.o HHLeakMechanism.cu.o HHNaCurrMechanism.cu.o \
	HHKCurrMechanism.cu.o DCCurrentMech.cu.o HHGradedGABAAMechanism.cu.o \
	HHSpikeGABAAMechanism.cu.o

###############################
#      Linker (LD) Flags      #
###############################

LD_FLAGS            := -L. -lm -lpthread -lrt
CUDART_LD_FLAGS     := -L$(CUDA_LIB_PATH) -lcudart
CUDA_BIN_LDFLAGS    := $(CUDART_LD_FLAGS) $(LD_FLAGS) -Wl,-rpath=$(CUDA_LIB_PATH)

###############################
#       Definition Flags      #
###############################

DEFINES ?=
ifdef CUDA
DEFINES += -DCUDA
endif
ifdef DEBUG
DEFINES += -DDEBUG
else
DEFINES += -DNDEBUG
endif

CUDA_DEFINES := $(DEFINES)
CUDA_BIN_DEFINES ?= $(DEFINES)

###############################
#    Include Path & Flags     #
###############################

CUDA_INCLUDES := -I$(CUDA_INC_PATH)
INCLUDES := $(CUDA_INCLUDES) -I.

###############################
#        Make Targets         #
###############################

SIMUL_MAIN := dsac
SIMUL_MAIN_OBJ := $(addsuffix .o, $(SIMUL_MAIN))
SIMUL_MAIN_BIN := $(addsuffix .bin, $(SIMUL_MAIN))

CUDA_LINK_OBJ := 
ifdef CUDA
CUDA_LINK_OBJ	+= dlink.o
endif

OBJECTS		:= $(wildcard *.o)
LIBRARIES	:= $(wildcard *.a)
BINARIES	:= $(wildcard *.bin)

###############################
#         Make Rules          #
###############################

# ------- Build Rules -------

.PHONY: clean all build run remake rebuild

build: all

all: $(SIMUL_MAIN_BIN)

run: $(SIMUL_MAIN_BIN)
	./$<

clean:
	@rm -f $(OBJECTS) $(LIBRARIES) $(BINARIES) *.s *.i *.ii

remake: clean build

rebuild: remake

# ------- CPU Myriad Library -------

$(MYRIAD_LIB_OBJS): %.c.o : %.c
	$(CC) $(CCFLAGS) $(INCLUDES) $(DEFINES) -o $@ -c $<

# ------- CUDA Myriad Library -------

$(CUDA_MYRIAD_LIB): $(CUDA_MYRIAD_LIB_OBJS)
	$(NVCC) -lib $^ -o $(CUDA_MYRIAD_LIB)

$(CUDA_MYRIAD_LIB_OBJS): %.cu.o : %.cu
	$(NVCC) $(NVCC_HOSTCC_FLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) \
	$(CUDA_INCLUDES) $(CUDA_DEFINES) -o $@ -dc $<

# ------- Linker Object -------

$(CUDA_LINK_OBJ): $(SIMUL_MAIN_OBJ) $(CUDA_MYRIAD_LIB_OBJS)
	$(NVCC) $(GENCODE_FLAGS) -dlink $^ -o $(CUDA_LINK_OBJ) # Necessary for seperate compilation

# ------- Main binary object -------

$(SIMUL_MAIN_OBJ): %.o : %.cu
ifdef CUDA
	$(NVCC) $(NVCC_HOSTCC_FLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) \
	$(CUDA_INCLUDES) $(CUDA_BIN_DEFINES) -o $@ -dc $<
else
	$(CC) $(CCFLAGS) $(INCLUDES) $(DEFINES) -x c -c $< -o $@
endif

# ------- Host Linker Generated Binary -------

$(SIMUL_MAIN_BIN): $(SIMUL_MAIN_OBJ) $(CUDA_LINK_OBJ) $(MYRIAD_LIB_OBJS) $(CUDA_MYRIAD_LIB_OBJS)
ifdef CUDA
	$(CC) -o $@ $+ $(CUDA_BIN_LDFLAGS)
else
	$(CC) -o $@ $+ -flto $(LD_FLAGS)
endif # CUDA


# ------- Doxygen Documentation Generation -------
doxygen:
	$(DOXYGEN) $(DOXYGEN_CONF)

# ------- Ctags Generation -------
ctags:
	$(CTAGS) $(CTAGS_FLAGS)

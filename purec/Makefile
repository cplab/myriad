###############################
#     CUDA Binaries & Libs    #
###############################
CUDA_PATH     ?= /usr/local/cuda
CUDA_INC_PATH ?= $(CUDA_PATH)/include
CUDA_BIN_PATH ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH ?= $(CUDA_PATH)/lib64

###############################
#      Compilers & Tools      #
###############################
NVCC ?= $(CUDA_BIN_PATH)/nvcc
CC   := x86_64-linux-gnu-gcc

###############################
#       COMPILER FLAGS        #
###############################

# OS Arch Flags
OS_SIZE = 64

# Common C flags shared by all
COMMON_CFLAGS := -Wall -Wextra -Wno-unused-parameter -Wno-unknown-pragmas
ifdef DEBUG
COMMON_CFLAGS += -g3
endif

# Flags only given to host-side C
CCFLAGS := $(COMMON_CFLAGS) -Wpedantic -std=gnu99 -fopenmp
ifndef DEBUG
CCFLAGS += -O2 -march=native -fno-trapping-math -ffinite-math-only -fno-math-errno -fpredictive-commoning -fprefetch-loop-arrays
ifdef LTO
CCFLAGS += -flto
endif # ifdef LTO
else
CCFLAGS += -Og
endif # ifndef DEBUG

# Flags given only to CUDA C
CUFLAGS := $(COMMON_CFLAGS) -m$(OS_SIZE) --device-c -x cu -ccbin $(CC) $(addprefix -Xcompiler , $(CUFLAGS))
ifdef DEBUG
CUFLAGS := -g -G
endif
GENCODE_FLAGS := -gencode arch=compute_35,code=sm_35

###############################
#        Libraries            #
###############################

# CPU Myriad Library
MYRIAD_LIB_OBJS := MyriadObject.c.o Mechanism.c.o Compartment.c.o \
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

LD_FLAGS     := -L. -lm -lpthread -lrt -lgomp
CUDA_LDFLAGS := -L$(CUDA_LIB_PATH) -lcudart $(LD_FLAGS) -Wl,-rpath=$(CUDA_LIB_PATH)

###############################
#       Definition Flags      #
###############################

#  -DFAST_EXP 
DEFINES ?= -DFAST_EXP -DNUM_THREADS=8

ifdef CUDA
DEFINES += -DCUDA
endif

ifdef DEBUG
DEFINES += -DDEBUG
else
DEFINES += -DNDEBUG
endif

CUDA_DEFINES := $(DEFINES)

###############################
#    Include Path & Flags     #
###############################

CUDA_INCLUDES := -I$(CUDA_INC_PATH)
INCLUDES := -I.

###############################
#        Make Targets         #
###############################

SIMUL_MAIN := dsac
SIMUL_MAIN_OBJ := $(addsuffix .o, $(SIMUL_MAIN))
SIMUL_MAIN_BIN := $(addsuffix .bin, $(SIMUL_MAIN))

CUDA_LINK_OBJ := 
ifdef CUDA
CUDA_LINK_OBJ += dlink.o
endif

OBJECTS     := $(wildcard *.o)
LIBRARIES   := $(wildcard *.a)
BINARIES    := $(wildcard *.bin)

###############################
#         Make Rules          #
###############################

# ------- Build Rules -------

.PHONY: clean cuda host

clean:
	@rm -f $(OBJECTS) $(LIBRARIES) $(BINARIES) *.s *.i *.ii

# ------- Host Myriad Objects -------

$(MYRIAD_LIB_OBJS): %.c.o : %.c
ifdef CUDA
	$(CC) $(CCFLAGS) $(INCLUDES) $(CUDA_INCLUDES) $(DEFINES) -o $@ -c $<
else
	$(CC) $(CCFLAGS) $(INCLUDES) $(DEFINES) -o $@ -c $<
endif 

# ------- CUDA Myriad Objects -------

$(CUDA_MYRIAD_LIB_OBJS): %.cu.o : %.cu
	$(NVCC) $(CUFLAGS) $(GENCODE_FLAGS) $(CUDA_INCLUDES) $(CUDA_DEFINES) -o $@ -dc $<

# ------- CUDA Linker Object -------

$(CUDA_LINK_OBJ): $(SIMUL_MAIN_OBJ) $(CUDA_MYRIAD_LIB_OBJS)
	$(NVCC) $(GENCODE_FLAGS) -dlink $^ -o $(CUDA_LINK_OBJ)

# ------- Main binary object -------

$(SIMUL_MAIN_OBJ): %.o : %.cu
ifdef CUDA
	$(NVCC) $(CUFLAGS) $(GENCODE_FLAGS) $(CUDA_INCLUDES) $(CUDA_BIN_DEFINES) -o $@ -dc $<
else
	$(CC) $(CCFLAGS) $(INCLUDES) $(DEFINES) -x c -c $< -o $@
endif

# ------- Host Linker Generated Binary -------

cuda: $(SIMUL_MAIN_OBJ) $(CUDA_LINK_OBJ) $(MYRIAD_LIB_OBJS) $(CUDA_MYRIAD_LIB_OBJS)
	$(CC) -o $@ $+ $(CUDA_LDFLAGS)

host: $(SIMUL_MAIN_OBJ) $(MYRIAD_LIB_OBJS)
ifdef LTO
	$(CC) -o $@.bin $+ -flto $(LD_FLAGS)
else
	$(CC) -o $@.bin $+ $(LD_FLAGS)
endif

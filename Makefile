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
NVCC	:= $(CUDA_BIN_PATH)/nvcc
CC	:= gcc
CPP	:= g++
AR	:= ar
CTAGS ?= ctags-exuberant

###############################
#       COMPILER FLAGS        #
###############################

# OS Arch Flags
OS_SIZE = 64
OS_ARCH = x86_64

# CC & related flags
CCOMMON_FLAGS	:= -g3 -O0 -Wall -Wpedantic
CCFLAGS		:= $(CCOMMON_FLAGS) -std=c99
CPPFLAGS	:= $(CCOMMON_FLAGS) -std=c++11
CUFLAGS		:= $(CCOMMON_FLAGS)

# NVCC & related flags
vNVCC_HOSTCC_FLAGS = -x cu -ccbin $(CC) $(addprefix -Xcompiler , $(CUFLAGS))
NVCCFLAGS := -m$(OS_SIZE) -g -G -pg
GENCODE_FLAGS := -gencode arch=compute_30,code=sm_30
EXTRA_NVCC_FLAGS := -rdc=true

###############################
#        Libraries            #
###############################

# Static Libraries

# CPU Myriad Library
MYRIAD_LIB_LDNAME 	:= myriad
MYRIAD_LIB 		:= lib$(MYRIAD_LIB_LDNAME).a
MYRIAD_LIB_OBJS 	:= myriad_debug.c.o MyriadObject.c.o Mechanism.c.o Compartment.c.o \
	HHSomaCompartment.c.o HHLeakMechanism.c.o

# CUDA Myriad Library
CUDA_MYRIAD_LIB_LDNAME	:= cudamyriad
CUDA_MYRIAD_LIB		:= 
CUDA_MYRIAD_LIB_OBJS	:= 

ifdef CUDA
CUDA_MYRIAD_LIB		:= lib$(CUDA_MYRIAD_LIB_LDNAME).a
CUDA_MYRIAD_LIB_OBJS	+= MyriadObject.cu.o Mechanism.cu.o Compartment.cu.o \
	HHSomaCompartment.cu.o HHLeakMechanism.cu.o
endif

# Shared Libraries

###############################
#      Linker (LD) Flags      #
###############################

LD_FLAGS 		:= -L. -l$(MYRIAD_LIB_LDNAME) -l$(CUDA_MYRIAD_LIB_LDNAME)
CUDART_LD_FLAGS		:= -L$(CUDA_LIB_PATH) -lcudart
CUDA_BIN_LDFLAGS	:= $(CUDART_LD_FLAGS) $(LD_FLAGS)

###############################
#       Definition Flags      #
###############################

CUDA_BIN_DEFINES ?= -DUNIT_TEST

###############################
#    Include Path & Flags     #
###############################

CUDA_INCLUDES := -I$(CUDA_INC_PATH)
INCLUDES := $(CUDA_INCLUDES) -I.

###############################
#        Make Targets         #
###############################

SIMUL_MAIN_OBJ := main.o
SIMUL_MAIN_BIN := main.bin

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
	-rm -f $(OBJECTS) $(LIBRARIES) $(BINARIES)

remake: clean build

rebuild: remake

# ------- CPU Myriad Library -------

$(MYRIAD_LIB): $(MYRIAD_LIB_OBJS)
	$(AR) rcs $@ $^

$(MYRIAD_LIB_OBJS): %.c.o : %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

# ------- CUDA Myriad Library -------

$(CUDA_MYRIAD_LIB): $(CUDA_MYRIAD_LIB_OBJS)
	$(NVCC) -lib $^ -o $(CUDA_MYRIAD_LIB)

$(CUDA_MYRIAD_LIB_OBJS): %.cu.o : %.cu
	$(NVCC) $(NVCC_HOSTCC_FLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) \
	$(CUDA_INCLUDES) -o $@ -dc $<

# ------- Linker Object -------

$(CUDA_LINK_OBJ): $(SIMUL_MAIN_OBJ) $(CUDA_MYRIAD_LIB)
	$(NVCC) $(GENCODE_FLAGS) -dlink $^ -o $(CUDA_LINK_OBJ) # Necessary for seperate compilation

# ------- Main binary object -------

$(SIMUL_MAIN_OBJ): %.o : %.cu
ifdef CUDA
	$(NVCC) $(NVCC_HOSTCC_FLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) \
	$(CUDA_INCLUDES) $(CUDA_BIN_DEFINES) -o $@ -dc $<
else
	$(CPP) $(CPPFLAGS) -x c++ -c $< -o $@
endif

# ------- Host Linker Generated Binary -------

$(SIMUL_MAIN_BIN): $(SIMUL_MAIN_OBJ) $(CUDA_LINK_OBJ) $(MYRIAD_LIB) $(CUDA_MYRIAD_LIB)
ifdef CUDA
	$(CC) -I. -o $@ $+ $(CUDA_BIN_LDFLAGS)
else
	$(CC) -I. -o $@ $+
endif

# ------- Bonus Ctags Generation -------
ctags:
	$(CTAGS) -e -R --langmap=c:.cu.cuh
#	$(CTAGS) --verbose -R --langmap=c:.cu.cuh --fields="+afikKlmnsSzt"

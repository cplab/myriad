
###############################
#     CUDA Binaries & Libs    #
###############################
CUDA_PATH		?= /usr/local/cuda
CUDA_INC_PATH	?= $(CUDA_PATH)/include
CUDA_BIN_PATH	?= $(CUDA_PATH)/bin
CUDA_LIB_PATH	?= $(CUDA_PATH)/lib64

###############################
#      Compilers & Tools      #
###############################
NVCC	:= $(CUDA_BIN_PATH)/nvcc
CC		:= gcc
CPP		:= g++
AR		:= ar

###############################
#       COMPILER FLAGS        #
###############################

# OS Arch Flags
OS_SIZE = 64
OS_ARCH = x86_64

# CC & related flags
CCOMMON_FLAGS	:= -g3 -O0 -Wall
CCFLAGS			:= $(CCOMMON_FLAGS) -std=c99
CPPFLAGS		:= $(CCOMMON_FLAGS) -fpermissive

# NVCC & related flags
NVCC_HOSTCC_FLAGS = -ccbin $(CPP) $(addprefix -Xcompiler , $(CPPFLAGS))
NVCCFLAGS := -m64 -g -G
GENCODE_FLAGS := -gencode arch=compute_30,code=sm_30
EXTRA_NVCC_FLAGS := -rdc=true

###############################
#  Libraries & Linker Flags   #
###############################

# Static Libraries

# CPU Myriad Library
MYRIAD_LIB_LDNAME 	:= myriad
MYRIAD_LIB 			:= lib$(MYRIAD_LIB_LDNAME).a
MYRIAD_LIB_OBJS 	:= myriad_debug.o
# CUDA Myriad Library
CUDA_MYRIAD_LIB_LDNAME	:= cudamyriad
CUDA_MYRIAD_LIB			:= lib$(CUDA_MYRIAD_LIB_LDNAME).a
CUDA_MYRIAD_LIB_OBJS	:= MyriadObject.o Mechanism.o
# CUDA_MYRIAD_LINK_OBJ:= CUDA_MyriadObject_Link.o

# LD Flags
LD_FLAGS 			:= -L. -l$(MYRIAD_LIB_LDNAME) -l$(CUDA_MYRIAD_LIB_LDNAME)
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

###############################
#        Make Targets         #
###############################

SIMUL_MAIN_OBJ := main.o
SIMUL_MAIN_BIN := main.bin

CUDA_LINK_OBJ := dlink.o

OBJECTS		:= $(wildcard *.o)
LIBRARIES	:= $(wildcard *.a)
BINARIES	:= $(wildcard *.bin)

###############################
#         Make Rules          #
###############################

.PHONY: clean all build run remake rebuild

build: all

all: $(SIMUL_MAIN_BIN)

# CPU Myriad Library
$(MYRIAD_LIB): $(MYRIAD_LIB_OBJS)
	$(AR) rcs $@ $^

$(MYRIAD_LIB_OBJS): %.o : %.c
	$(CC) $(CCFLAGS) -o $@ -c $<
###################################


# CUDA Myriad Library
$(CUDA_MYRIAD_LIB): $(CUDA_MYRIAD_LIB_OBJS)
	$(NVCC) -lib $^ -o $(CUDA_MYRIAD_LIB)

$(CUDA_MYRIAD_LIB_OBJS): %.o : %.cu
	$(NVCC) $(NVCC_HOSTCC_FLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(CUDA_INCLUDES) -o $@ -dc $<

###################################

# Linker object necessary for main binary seperate compilation
$(CUDA_LINK_OBJ): $(SIMUL_MAIN_OBJ) $(CUDA_MYRIAD_LIB)
	$(NVCC) $(GENCODE_FLAGS) -dlink $^ -o $(CUDA_LINK_OBJ)

# Main binary object
$(SIMUL_MAIN_OBJ): %.o : %.cu
	$(NVCC) $(NVCC_HOSTCC_FLAGS) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(CUDA_INCLUDES) $(CUDA_BIN_DEFINES) -o $@ -dc $<

# Can use host linker to generate binary instead of nvcc
$(SIMUL_MAIN_BIN): $(SIMUL_MAIN_OBJ) $(CUDA_LINK_OBJ) $(MYRIAD_LIB) $(CUDA_MYRIAD_LIB)
	$(CC) -o $@ $+ $(CUDA_BIN_LDFLAGS)

run: $(SIMUL_MAIN_BIN)
	./$<

clean:
	-rm -f $(OBJECTS) $(LIBRARIES) $(BINARIES)

remake: clean build

rebuild: remake

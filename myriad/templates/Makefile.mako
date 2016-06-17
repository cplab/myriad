##########################
## CUDA Binaries & Libs ##
##########################

CUDA_PATH     ?= /usr
CUDA_INC_PATH ?= $(CUDA_PATH)/include
CUDA_BIN_PATH ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH ?= $(CUDA_PATH)/lib/x86_64-linux-gnu/

#######################
## Compilers & Tools ##
#######################

NVCC ?= $(CUDA_BIN_PATH)/nvcc
CC   := gcc-4.9
CXX  := g++-4.9

####################
## Compiler Flags ##
####################

## Enable link-time optimization (requires GCC 4.8+ and gold linker)
ifdef LTO
FLTO := -flto
endif
## Enable undefined behaviour sanitizer (requires GCC 4.9+)
% if DEBUG:
UBSAN := -fsanitize=undefined
% endif

## Flags only given to host-side C
CCFLAGS := -Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-unknown-pragmas -std=gnu99 -fopenmp
% if DEBUG:
CCFLAGS += -Og -g3 $(UBSAN)
% else:
CCFLAGS += -O2 $(FLTO) -march=native -fno-trapping-math -ffinite-math-only -fno-math-errno -fpredictive-commoning -fprefetch-loop-arrays
% endif

## Flags given only to CUDA C
CUFLAGS := -m64 -x cu -ccbin $(CXX)
% if DEBUG:
CUFLAGS += -g -G
% endif
## GENCODE_FLAGS := -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS := -arch=sm_35

##################
## Linker Flags ##
##################

LDFLAGS     := -L. -lm -lpthread -lrt -lgomp
CUDA_LDFLAGS := -L$(CUDA_LIB_PATH) -lcudart -lcudadevrt

######################
## Definition Flags ##
######################

DEFINES ?= -DFAST_EXP -DNUM_THREADS=${NUM_THREADS}

## CUDA 7.5 needs FORCE_INLINES to get around some kind of issue
% if CUDA:
DEFINES += -DCUDA -D_FORCE_INLINES
% endif

% if DEBUG:
DEFINES += -DDEBUG
% else:
DEFINES += -DNDEBUG
% endif

##################
## Include Path ##
##################

CUDA_INCLUDES := -I$(CUDA_INC_PATH)
INCLUDES := -I.

##################
## Make Targets ##
##################

% if CUDA:
CUDA_LINK_OBJ := dlink.o
% endif
OBJECTS := ${myriad_lib_objs}
COBJECTS := myriad_alloc.o myriad_communicator.o
BINARY  := app.bin

################
## Make Rules ##
################

.PHONY: clean all

all: $(BINARY)

clean:
	@rm -f *.bin *.o *.s *.i *.ii

## ------- Myriad Objects -------

$(COBJECTS): %.o : %.c
	$(CC) $(DEFINES) -x c $(CCFLAGS) $(INCLUDES) $(DEFINES) -o $@ -c $<

$(OBJECTS): %.o : %.cu
% if CUDA:
	$(NVCC) $(DEFINES) $(GENCODE_FLAGS) $(CUFLAGS) $(CUDA_INCLUDES) -o $@ -dc $<
% else:
	$(CC) $(DEFINES) -x c $(CCFLAGS) $(INCLUDES) $(DEFINES) -o $@ -c $<
% endif

## ------- CUDA Linker Object -------

$(CUDA_LINK_OBJ): $(OBJECTS)
% if CUDA:
	$(NVCC) $(GENCODE_FLAGS) -dlink $^ -o $@ $(CUDA_LDFLAGS)
% endif

## ------- Final Binary -------

$(BINARY): $(OBJECTS) $(COBJECTS) $(CUDA_LINK_OBJ)
% if CUDA:
	$(NVCC) -o $@ $+ $(CUDA_LDFLAGS)
% else:
	$(CC) $(UBSAN) $(FLTO) -o $@ $+ $(LDFLAGS)
% endif

## ----- Python build -----

python_build:
	python setup.py build
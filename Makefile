.PHONY: all clean

CC = gcc
CFLAGS = -Wall -O0 -g3 -std=c99
LDFLAGS = -static -L. -lmyriad

OBJECTS = Object.o Point.o Circle.o Compartment.o Mechanism.o LifCompartment.o DCMechanism.o LifLeakMechanism.o
LIBRARIES = libmyriad.a
BINARIES = any_test points_test circles_test compartments

OBJDIR = obj
LIBDIR = lib
BINDIR = bin

export

$(OBJDIR):
	mkdir $(OBJDIR)

$(LIBDIR):
	mkdir $(LIBDIR)
	
$(BINDIR):
	mkdir $(BINDIR)

all: $(OBJDIR) $(LIBDIR) $(BINDIR)
	cd src; make all

clean:
	@rm -rf $(OBJDIR) $(LIBDIR) $(BINDIR)
CC = gcc
CFLAGS = -Wall -O0 -g3 -std=c99
LDFLAGS = -static -L. -lmyriad

OBJECTS = Object.o Point.o Circle.o Compartment.o Mechanism.o LifCompartment.o DCMechanism.o LifLeakMechanism.o

Object.o:
	$(CC) $(CFLAGS) -c Object.c -o Object.o

Point.o: Object.o
	$(CC) $(CFLAGS) -c Point.c -o Point.o
	
Circle.o: Object.o Point.o
	$(CC) $(CFLAGS) -c Circle.c -o Circle.o

Compartment.o: Object.o
	$(CC) $(CFLAGS) -c Compartment.c -o Compartment.o

Mechanism.o: Object.o
	$(CC) $(CFLAGS) -c Mechanism.c -o Mechanism.o

DCMechanism: Mechanism.o
	$(CC) $(CFLAGS) -c DCMechanism.c -o DCMechanism.o

LifCompartment.o: Compartment.o
	$(CC) $(CFLAGS) -c LifCompartment.c -o LifCompartment.o

LifLeakMechanism.o: Mechanism.o LifCompartment.o
	$(CC) $(CFLAGS) -c LifLeakMechanism.c -o LifLeakMechanism.o

libmyriad: $(OBJECTS)
	ar rcs libmyriad.a $(OBJECTS)

all: $(OBJECTS) libmyriad
	$(CC) $(CFLAGS) any.c $(LDFLAGS) -o any_test
	$(CC) $(CFLAGS) points.c $(LDFLAGS) -o points_test
	$(CC) $(CFLAGS) circles.c $(LDFLAGS) -o circles_test
	$(CC) $(CFLAGS) compartments.c  $(LDFLAGS) -o compartments 

clean:
	rm -f *.o libmyriad.a any_test circles_test points_test compartments
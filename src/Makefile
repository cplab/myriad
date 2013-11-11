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

libmyriad.a: $(OBJECTS)
	ar rcs libmyriad.a $(OBJECTS)
	
any_test: $(OBJECTS)
	$(CC) $(CFLAGS) any.c $(LDFLAGS) -o any_test
	
points_test: $(OBJECT)
	$(CC) $(CFLAGS) points.c $(LDFLAGS) -o points_test

circles_test: $(OBJECT)
	$(CC) $(CFLAGS) circles.c $(LDFLAGS) -o circles_test
	
compartments: $(OBJECT)
	$(CC) $(CFLAGS) compartments.c  $(LDFLAGS) -o compartments

all: $(OBJECTS) $(LIBRARIES) $(BINARIES)
	mv $(OBJECTS) ../$(OBJDIR)
	mv $(LIBRARIES) ../$(LIBDIR)
	mv $(BINARIES) ../$(BINDIR)
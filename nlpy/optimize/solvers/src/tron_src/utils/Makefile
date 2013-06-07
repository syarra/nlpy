#
# UTILS library.
#

L_ARCH   = $(ARCH)
LIB_NAME = d-$(L_ARCH).a

CFLAGS    = $(OPTFLAGS) -D$(ARCH)
FFLAGS    = $(OPTFLAGS)
CPPFLAGS  = -D$(ARCH)

AR = ar -rcv

FILES = cputime.o dpmeps.o wallclock.o

src: $(FILES)
	$(AR) $(LIB_NAME) $(FILES)

clean:
	- rm -f *.o

.SUFFIXES: .c .f .F

.c.o:
	$(CC) $(CFLAGS) -c $*.c
.f.o:
	$(FC) $(FFLAGS) -c $*.f
.F.o:
#	./Fpp $*.F $(CPPFLAGS)
	$(FC) $(FFLAGS) -c $*.F
#	rm $*.f

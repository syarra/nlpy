#
# BLAS directory.
#

L_ARCH   = $(ARCH)
LIB_NAME = d-$(L_ARCH).a

CFLAGS = $(OPTFLAGS) 
FFLAGS = $(OPTFLAGS)

AR = ar -rcv

FILES = daxpy.o    dcopy.o    ddot.o     dnrm2.o    dscal.o    idamax.o
        
blas: $(FILES) 
	$(AR) $(LIB_NAME) $(FILES)  

clean:
	- rm -f *.o
	- rm -f *.a

.c.o:
	$(CC) $(CFLAGS) -c $*.c
.f.o:
	$(FC) $(FFLAGS) -c $*.f



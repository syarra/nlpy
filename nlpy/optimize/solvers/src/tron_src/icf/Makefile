#
# ICF directory.
#

L_ARCH   = $(ARCH)
LIB_NAME = d-$(L_ARCH).a

CFLAGS = $(OPTFLAGS) 
FFLAGS = $(OPTFLAGS)

AR = ar -rcv

FILES = dicf.o      dpcg.o      dssyax.o    ihsort.o    srtdat2.o \
        dicfs.o     dsel2.o     dstrsol.o   insort.o

icf: $(FILES) 
	$(AR) $(LIB_NAME) $(FILES)  

clean:
	- rm -f *.o
	- rm -f *.a

.c.o:
	$(CC) $(CFLAGS) -c $*.c
.f.o:
	$(FC) $(FFLAGS) -c $*.f



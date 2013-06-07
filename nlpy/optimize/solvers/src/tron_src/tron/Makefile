#
# TRON directory.
#

L_ARCH   = $(ARCH)
LIB_NAME = d-$(L_ARCH).a

CFLAGS = $(OPTFLAGS) 
FFLAGS = $(OPTFLAGS)

AR = ar -rcv

FILES = dbreakpt.o   dgpstep.o    dsetsp.o     dtron.o  \
        dcauchy.o    dmid.o       dspcg.o      dtrpcg.o \
        dgpnrm2.o    dprsrch.o    dsphesd.o    dtrqsol.o \
        asubprod.o

src: $(FILES) 
	$(AR) $(LIB_NAME) $(FILES)  

clean:
	- rm -f *.o
	- rm -f *.a

.c.o:
	$(CC) $(CFLAGS) -c $*.c
.f.o:
	$(FC) $(FFLAGS) -c $*.f



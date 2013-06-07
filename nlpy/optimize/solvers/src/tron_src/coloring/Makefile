#
# COLORING directory.
#

L_ARCH   = $(ARCH)
LIB_NAME = d-$(L_ARCH).a

CFLAGS = $(OPTFLAGS) 
FFLAGS = $(OPTFLAGS)

AR = ar -rcv

FILES = degr.o     ido.o      numsrt.o   seq.o      slo.o      srtdat.o \
        dssm.o     idog.o     sdpt.o     setr.o     slog.o

coloring: $(FILES) 
	$(AR) $(LIB_NAME) $(FILES)  

clean:
	- rm -f *.o
	- rm -f *.a

.c.o:
	$(CC) $(CFLAGS) -c $*.c
.f.o:
	$(FC) $(FFLAGS) -c $*.f



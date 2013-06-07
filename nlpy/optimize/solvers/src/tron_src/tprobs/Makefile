#
# TPROBS directory.
#

L_ARCH   = $(ARCH)
LIB_NAME = d-$(L_ARCH).a

CFLAGS = $(OPTFLAGS) 
FFLAGS = $(OPTFLAGS)

AR = ar -rcv

FILES = deptfg.o   dgl1sp.o   dljcfg.o   dmsabc.o   dodchs.o   dpjbsp.o \
        depths.o   dgl2co.o   dminfg.o   dmsafg.o   dodcps.o   dsscfg.o \
        deptsp.o   dgl2fg.o   dminhs.o   dmsahs.o   dodcsp.o   dsschs.o \
        dgl1fg.o   dgl2hs.o   dminsp.o   dmsasp.o   dpjbfg.o   dsscsp.o \
        dgl1hs.o   dgl2sp.o   dminxb.o   dodcfg.o   dpjbhs.o

tprobs: $(FILES) 
	$(AR) $(LIB_NAME) $(FILES)  

clean:
	- rm -f *.o
	- rm -f *.a

.c.o:
	$(CC) $(CFLAGS) -c $*.c
.f.o:
	$(FC) $(FFLAGS) -c $*.f



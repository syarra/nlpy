      subroutine seq(n,indrow,jpntr,indcol,ipntr,list,ngrp,maxgrp,
     *               iwa)
      integer n,maxgrp
      integer indrow(*),jpntr(n+1),indcol(*),ipntr(*),list(n),
     *        ngrp(n),iwa(n)
c     **********
c
c     subroutine seq
c
c     Given the sparsity pattern of an m by n matrix A, this
c     subroutine determines a consistent partition of the
c     columns of A by a sequential algorithm.
c
c     A consistent partition is defined in terms of the loopless
c     graph G with vertices a(j), j = 1,2,...,n where a(j) is the
c     j-th column of A and with edge (a(i),a(j)) if and only if
c     columns i and j have a non-zero in the same row position.
c
c     A partition of the columns of A into groups is consistent
c     if the columns in any group are not adjacent in the graph G.
c     In graph-theory terminology, a consistent partition of the
c     columns of A corresponds to a coloring of the graph G.
c
c     The subroutine examines the columns in the order specified
c     by the array list, and assigns the current column to the
c     group with the smallest possible number.
c
c     Note that the value of m is not needed by seq and is
c     therefore not present in the subroutine statement.
c
c     The subroutine statement is
c
c       subroutine seq(n,indrow,jpntr,indcol,ipntr,list,ngrp,maxgrp,
c                      iwa)
c
c     where
c
c       n is a positive integer input variable set to the number
c         of columns of A.
c
c       indrow is an integer input array which contains the row
c         indices for the non-zeroes in the matrix A.
c
c       jpntr is an integer input array of length n + 1 which
c         specifies the locations of the row indices in indrow.
c         The row indices for column j are
c
c               indrow(k), k = jpntr(j),...,jpntr(j+1)-1.
c
c         Note that jpntr(n+1)-1 is then the number of non-zero
c         elements of the matrix A.
c
c       indcol is an integer input array which contains the
c         column indices for the non-zeroes in the matrix A.
c
c       ipntr is an integer input array of length m + 1 which
c         specifies the locations of the column indices in indcol.
c         The column indices for row i are
c
c               indcol(k), k = ipntr(i),...,ipntr(i+1)-1.
c
c         Note that ipntr(m+1)-1 is then the number of non-zero
c         elements of the matrix A.
c
c       list is an integer input array of length n which specifies
c         the order to be used by the sequential algorithm.
c         The j-th column in this order is list(j).
c
c       ngrp is an integer output array of length n which specifies
c         the partition of the columns of A. Column jcol belongs
c         to group ngrp(jcol).
c
c       maxgrp is an integer output variable which specifies the
c         number of groups in the partition of the columns of A.
c
c       iwa is an integer work array of length n.
c
c     Argonne National Laboratory. MINPACK Project. July 1983.
c     Thomas F. Coleman, Burton S. Garbow, Jorge J. More'
c
c     **********
      integer ic,ip,ir,j,jcol,jp
c
c     Initialization block.
c
      maxgrp = 0
      do 10 jp = 1, n
         ngrp(jp) = n
         iwa(jp) = 0
   10    continue
c
c     Beginning of iteration loop.
c
      do 60 j = 1, n
         jcol = list(j)
c
c        Find all columns adjacent to column jcol.
c
c        Determine all positions (ir,jcol) which correspond
c        to non-zeroes in the matrix.
c
         do 30 jp = jpntr(jcol), jpntr(jcol+1)-1
            ir = indrow(jp)
c
c           For each row ir, determine all positions (ir,ic)
c           which correspond to non-zeroes in the matrix.
c
            do 20 ip = ipntr(ir), ipntr(ir+1)-1
               ic = indcol(ip)
c
c              Array iwa marks the group numbers of the
c              columns which are adjacent to column jcol.
c
               iwa(ngrp(ic)) = j
   20          continue
   30       continue
c
c        Assign the smallest un-marked group number to jcol.
c
         do 40 jp = 1, maxgrp
            if (iwa(jp) .ne. j) go to 50
   40       continue
         maxgrp = maxgrp + 1
   50    continue
         ngrp(jcol) = jp
   60    continue
c
c        End of iteration loop.
c
      return
c
c     Last card of subroutine seq.
c
      end

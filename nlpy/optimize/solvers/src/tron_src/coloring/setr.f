      subroutine setr(m,n,indrow,jpntr,indcol,ipntr,iwa)
      integer m,n
      integer indrow(*),jpntr(n+1),indcol(*),ipntr(m+1),iwa(m)
c     **********
c
c     subroutine setr
c
c     Given a column-oriented definition of the sparsity pattern
c     of an m by n matrix A, this subroutine determines a
c     row-oriented definition of the sparsity pattern of A.
c
c     On input the column-oriented definition is specified by
c     the arrays indrow and jpntr. On output the row-oriented
c     definition is specified by the arrays indcol and ipntr.
c
c     The subroutine statement is
c
c       subroutine setr(m,n,indrow,jpntr,indcol,ipntr,iwa)
c
c     where
c
c       m is a positive integer input variable set to the number
c         of rows of A.
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
c       indcol is an integer output array which contains the
c         column indices for the non-zeroes in the matrix A.
c
c       ipntr is an integer output array of length m + 1 which
c         specifies the locations of the column indices in indcol.
c         The column indices for row i are
c
c               indcol(k), k = ipntr(i),...,ipntr(i+1)-1.
c
c         Note that ipntr(1) is set to 1 and that ipntr(m+1)-1 is
c         then the number of non-zero elements of the matrix A.
c
c       iwa is an integer work array of length m.
c
c     Argonne National Laboratory. MINPACK Project. July 1983.
c     Thomas F. Coleman, Burton S. Garbow, Jorge J. More'
c
c     **********
      integer ir,jcol,jp
c
c     Store in array iwa the counts of non-zeroes in the rows.
c
      do 10 ir = 1, m
         iwa(ir) = 0
   10    continue
      do 20 jp = 1, jpntr(n+1)-1
         iwa(indrow(jp)) = iwa(indrow(jp)) + 1
   20    continue
c
c     Set pointers to the start of the rows in indcol.
c
      ipntr(1) = 1
      do 30 ir = 1, m
         ipntr(ir+1) = ipntr(ir) + iwa(ir)
         iwa(ir) = ipntr(ir)
   30    continue
c
c     Fill indcol.
c
      do 50 jcol = 1, n
         do 40 jp = jpntr(jcol), jpntr(jcol+1)-1
            ir = indrow(jp)
            indcol(iwa(ir)) = jcol
            iwa(ir) = iwa(ir) + 1
   40       continue
   50    continue
      return
c
c     Last card of subroutine setr.
c
      end

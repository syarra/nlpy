      subroutine degr(n,indrow,jpntr,indcol,ipntr,ndeg,iwa)
      integer n
      integer indrow(*),jpntr(n+1),indcol(*),ipntr(*),ndeg(n),iwa(n)
c     **********
c
c     subroutine degr
c
c     Given the sparsity pattern of an m by n matrix A,
c     this subroutine determines the degree sequence for
c     the intersection graph of the columns of A.
c
c     In graph-theory terminology, the intersection graph of
c     the columns of A is the loopless graph G with vertices
c     a(j), j = 1,2,...,n where a(j) is the j-th column of A
c     and with edge (a(i),a(j)) if and only if columns i and j
c     have a non-zero in the same row position.
c
c     Note that the value of m is not needed by degr and is
c     therefore not present in the subroutine statement.
c
c     The subroutine statement is
c
c       subroutine degr(n,indrow,jpntr,indcol,ipntr,ndeg,iwa)
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
c       ndeg is an integer output array of length n which
c         specifies the degree sequence. The degree of the
c         j-th column of A is ndeg(j).
c
c       iwa is an integer work array of length n.
c
c     Argonne National Laboratory. MINPACK Project. July 1983.
c     Thomas F. Coleman, Burton S. Garbow, Jorge J. More'
c
c     **********
      integer ic,ip,ir,jcol,jp
c
c     Initialization block.
c
      do 10 jp = 1, n
         ndeg(jp) = 0
         iwa(jp) = 0
   10    continue
c
c     Compute the degree sequence by determining the contributions
c     to the degrees from the current(jcol) column and further
c     columns which have not yet been considered.
c
      do 40 jcol = 2, n
         iwa(jcol) = n
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
c              Array iwa marks columns which have contributed to
c              the degree count of column jcol. Update the degree
c              counts of these columns as well as column jcol.
c
               if (iwa(ic) .lt. jcol) then
                  iwa(ic) = jcol
                  ndeg(ic) = ndeg(ic) + 1
                  ndeg(jcol) = ndeg(jcol) + 1
                  end if
   20          continue
   30       continue
   40    continue
      return
c
c     Last card of subroutine degr.
c
      end

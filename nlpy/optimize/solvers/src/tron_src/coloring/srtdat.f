      subroutine srtdat(n,nnz,indrow,indcol,jpntr,iwa)
      integer n,nnz
      integer indrow(nnz),indcol(nnz),jpntr(n+1),iwa(n)
c     **********
c
c     subroutine srtdat
c
c     Given the non-zero elements of an m by n matrix A in
c     arbitrary order as specified by their row and column
c     indices, this subroutine permutes these elements so
c     that their column indices are in non-decreasing order.
c
c     On input it is assumed that the elements are specified in
c
c           indrow(k),indcol(k), k = 1,...,nnz.
c
c     On output the elements are permuted so that indcol is
c     in non-decreasing order. In addition, the array jpntr
c     is set so that the row indices for column j are
c
c           indrow(k), k = jpntr(j),...,jpntr(j+1)-1.
c
c     Note that the value of m is not needed by srtdat and is
c     therefore not present in the subroutine statement.
c
c     The subroutine statement is
c
c       subroutine srtdat(n,nnz,indrow,indcol,jpntr,iwa)
c
c     where
c
c       n is a positive integer input variable set to the number
c         of columns of A.
c
c       nnz is a positive integer input variable set to the number
c         of non-zero elements of A.
c
c       indrow is an integer array of length nnz. On input indrow
c         must contain the row indices of the non-zero elements of A.
c         On output indrow is permuted so that the corresponding
c         column indices of indcol are in non-decreasing order.
c
c       indcol is an integer array of length nnz. On input indcol
c         must contain the column indices of the non-zero elements
c         of A. On output indcol is permuted so that these indices
c         are in non-decreasing order.
c
c       jpntr is an integer output array of length n + 1 which
c         specifies the locations of the row indices in the output
c         indrow. The row indices for column j are
c
c               indrow(k), k = jpntr(j),...,jpntr(j+1)-1.
c
c         Note that jpntr(1) is set to 1 and that jpntr(n+1)-1
c         is then nnz.
c
c       iwa is an integer work array of length n.
c
c     Subprograms called
c
c       FORTRAN-supplied ... max
c
c     Argonne National Laboratory. MINPACK Project. July 1983.
c     Thomas F. Coleman, Burton S. Garbow, Jorge J. More'
c
c     **********
      integer i,j,k,l
c
c     Store in array iwa the counts of non-zeroes in the columns.
c
      do 10 j = 1, n
         iwa(j) = 0
   10    continue
      do 20 k = 1, nnz
         iwa(indcol(k)) = iwa(indcol(k)) + 1
   20    continue
c
c     Set pointers to the start of the columns in indrow.
c
      jpntr(1) = 1
      do 30 j = 1, n
         jpntr(j+1) = jpntr(j) + iwa(j)
         iwa(j) = jpntr(j)
   30    continue
      k = 1
c
c     Begin in-place sort.
c
   40 continue
         j = indcol(k)
         if (k .ge. jpntr(j)) then
c
c           Current element is in position. Now examine the
c           next element or the first un-sorted element in
c           the j-th group.
c
            k = max(k+1,iwa(j))
         else
c
c           Current element is not in position. Place element
c           in position and make the displaced element the
c           current element.
c
            l = iwa(j)
            iwa(j) = iwa(j) + 1
            i = indrow(k)
            indrow(k) = indrow(l)
            indcol(k) = indcol(l)
            indrow(l) = i
            indcol(l) = j
            end if
         if (k .le. nnz) go to 40
      return
c
c     Last card of subroutine srtdat.
c
      end

      subroutine slo(n,indrow,jpntr,indcol,ipntr,ndeg,list,
     *               maxclq,iwa1,iwa2,iwa3,iwa4)
      integer n,maxclq
      integer indrow(*),jpntr(n+1),indcol(*),ipntr(*),ndeg(n),
     *        list(n),iwa1(0:n-1),iwa2(n),iwa3(n),iwa4(n)
c     **********
c
c     subroutine slo
c
c     Given the sparsity pattern of an m by n matrix A, this
c     subroutine determines the smallest-last ordering of the
c     columns of A.
c
c     The smallest-last ordering is defined for the loopless
c     graph G with vertices a(j), j = 1,2,...,n where a(j) is the
c     j-th column of A and with edge (a(i),a(j)) if and only if
c     columns i and j have a non-zero in the same row position.
c
c     The smallest-last ordering is determined recursively by
c     letting list(k), k = n,...,1 be a column with least degree
c     in the subgraph spanned by the un-ordered columns.
c
c     Note that the value of m is not needed by slo and is
c     therefore not present in the subroutine statement.
c
c     The subroutine statement is
c
c       subroutine slo(n,indrow,jpntr,indcol,ipntr,ndeg,list,
c                      maxclq,iwa1,iwa2,iwa3,iwa4)
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
c       ndeg is an integer input array of length n which specifies
c         the degree sequence. The degree of the j-th column
c         of A is ndeg(j).
c
c       list is an integer output array of length n which specifies
c         the smallest-last ordering of the columns of A. The j-th
c         column in this order is list(j).
c
c       maxclq is an integer output variable set to the size
c         of the largest clique found during the ordering.
c
c       iwa1,iwa2,iwa3, and iwa4 are integer work arrays of length n.
c
c     Subprograms called
c
c       FORTRAN-supplied ... min
c
c     Argonne National Laboratory. MINPACK Project. August 1984.
c     Thomas F. Coleman, Burton S. Garbow, Jorge J. More'
c
c     **********
      integer ic,ip,ir,jcol,jp,mindeg,numdeg,numord
c
c     Initialization block.
c
      mindeg = n
      do 10 jp = 1, n
         iwa1(jp-1) = 0
         iwa4(jp) = n
         list(jp) = ndeg(jp)
         mindeg = min(mindeg,ndeg(jp))
   10    continue
c
c     Create a doubly-linked list to access the degrees of the
c     columns. The pointers for the linked list are as follows.
c
c     Each un-ordered column ic is in a list (the degree list)
c     of columns with the same degree.
c
c     iwa1(numdeg) is the first column in the numdeg list
c     unless iwa1(numdeg) = 0. In this case there are
c     no columns in the numdeg list.
c
c     iwa2(ic) is the column before ic in the degree list
c     unless iwa2(ic) = 0. In this case ic is the first
c     column in this degree list.
c
c     iwa3(ic) is the column after ic in the degree list
c     unless iwa3(ic) = 0. In this case ic is the last
c     column in this degree list.
c
c     If ic is an un-ordered column, then list(ic) is the
c     degree of ic in the graph induced by the un-ordered
c     columns. If jcol is an ordered column, then list(jcol)
c     is the smallest-last order of column jcol.
c
      do 20 jp = 1, n
         numdeg = ndeg(jp)
         iwa2(jp) = 0
         iwa3(jp) = iwa1(numdeg)
         if (iwa1(numdeg) .gt. 0) iwa2(iwa1(numdeg)) = jp
         iwa1(numdeg) = jp
   20    continue
      maxclq = 0
      numord = n
c
c     Beginning of iteration loop.
c
   30 continue
c
c        Choose a column jcol of minimal degree mindeg.
c
   40    continue
            jcol = iwa1(mindeg)
            if (jcol .gt. 0) go to 50
            mindeg = mindeg + 1
            go to 40
   50    continue
         list(jcol) = numord
c
c        Mark the size of the largest clique
c        found during the ordering.
c
         if (mindeg+1 .eq. numord .and. maxclq .eq. 0)
     *       maxclq = numord
c
c        Termination test.
c
         numord = numord - 1
         if (numord .eq. 0) go to 80
c
c        Delete column jcol from the mindeg list.
c
         iwa1(mindeg) = iwa3(jcol)
         if (iwa3(jcol) .gt. 0) iwa2(iwa3(jcol)) = 0
c
c        Find all columns adjacent to column jcol.
c
         iwa4(jcol) = 0
c
c        Determine all positions (ir,jcol) which correspond
c        to non-zeroes in the matrix.
c
         do 70 jp = jpntr(jcol), jpntr(jcol+1)-1
            ir = indrow(jp)
c
c           For each row ir, determine all positions (ir,ic)
c           which correspond to non-zeroes in the matrix.
c
            do 60 ip = ipntr(ir), ipntr(ir+1)-1
               ic = indcol(ip)
c
c              Array iwa4 marks columns which are adjacent to
c              column jcol.
c
               if (iwa4(ic) .gt. numord) then
                  iwa4(ic) = numord
c
c                 Update the pointers to the current degree lists.
c
                  numdeg = list(ic)
                  list(ic) = list(ic) - 1
                  mindeg = min(mindeg,list(ic))
c
c                 Delete column ic from the numdeg list.
c
                  if (iwa2(ic) .eq. 0) then
                     iwa1(numdeg) = iwa3(ic)
                  else
                     iwa3(iwa2(ic)) = iwa3(ic)
                     end if
                  if (iwa3(ic) .gt. 0) iwa2(iwa3(ic)) = iwa2(ic)
c
c                 Add column ic to the numdeg-1 list.
c
                  iwa2(ic) = 0
                  iwa3(ic) = iwa1(numdeg-1)
                  if (iwa1(numdeg-1) .gt. 0) iwa2(iwa1(numdeg-1)) = ic
                  iwa1(numdeg-1) = ic
                  end if
   60          continue
   70       continue
c
c        End of iteration loop.
c
         go to 30
   80 continue
c
c     Invert the array list.
c
      do 90 jcol = 1, n
         iwa2(list(jcol)) = jcol
   90    continue
      do 100 jp = 1, n
         list(jp) = iwa2(jp)
  100    continue
      return
c
c     Last card of subroutine slo.
c
      end

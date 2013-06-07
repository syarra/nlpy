      subroutine ido(m,n,indrow,jpntr,indcol,ipntr,ndeg,list,
     *               maxclq,iwa1,iwa2,iwa3,iwa4)
      integer m,n,maxclq
      integer indrow(*),jpntr(n+1),indcol(*),ipntr(m+1),ndeg(n),
     *        list(n),iwa1(0:n-1),iwa2(n),iwa3(n),iwa4(n)
c     **********
c
c     subroutine ido
c
c     Given the sparsity pattern of an m by n matrix A, this
c     subroutine determines an incidence-degree ordering of the
c     columns of A.
c
c     The incidence-degree ordering is defined for the loopless
c     graph G with vertices a(j), j = 1,2,...,n where a(j) is the
c     j-th column of A and with edge (a(i),a(j)) if and only if
c     columns i and j have a non-zero in the same row position.
c
c     The incidence-degree ordering is determined recursively by
c     letting list(k), k = 1,...,n be a column with maximal
c     incidence to the subgraph spanned by the ordered columns.
c     Among all the columns of maximal incidence, ido chooses a
c     column of maximal degree.
c
c     The subroutine statement is
c
c       subroutine ido(m,n,indrow,jpntr,indcol,ipntr,ndeg,list,
c                      maxclq,iwa1,iwa2,iwa3,iwa4)
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
c         the incidence-degree ordering of the columns of A. The j-th
c         column in this order is list(j).
c
c       maxclq is an integer output variable set to the size
c         of the largest clique found during the ordering.
c
c       iwa1,iwa2,iwa3, and iwa4 are integer work arrays of length n.
c
c     Subprograms called
c
c       MINPACK-supplied ... numsrt
c
c       FORTRAN-supplied ... max
c
c     Argonne National Laboratory. MINPACK Project. August 1984.
c     Thomas F. Coleman, Burton S. Garbow, Jorge J. More'
c
c     **********
      integer ic,ip,ir,jcol,jp,
     *        maxinc,maxlst,ncomp,numinc,numlst,numord,numwgt
c
c     Sort the degree sequence.
c
      call numsrt(n,n-1,ndeg,-1,iwa4,iwa2,iwa3)
c
c     Initialization block.
c
c     Create a doubly-linked list to access the incidences of the
c     columns. The pointers for the linked list are as follows.
c
c     Each un-ordered column ic is in a list (the incidence list)
c     of columns with the same incidence.
c
c     iwa1(numinc) is the first column in the numinc list
c     unless iwa1(numinc) = 0. In this case there are
c     no columns in the numinc list.
c
c     iwa2(ic) is the column before ic in the incidence list
c     unless iwa2(ic) = 0. In this case ic is the first
c     column in this incidence list.
c
c     iwa3(ic) is the column after ic in the incidence list
c     unless iwa3(ic) = 0. In this case ic is the last
c     column in this incidence list.
c
c     If ic is an un-ordered column, then list(ic) is the
c     incidence of ic to the graph induced by the ordered
c     columns. If jcol is an ordered column, then list(jcol)
c     is the incidence-degree order of column jcol.
c
      maxinc = 0
      do 10 jp = n, 1, -1
         ic = iwa4(jp)
         iwa1(n-jp) = 0
         iwa2(ic) = 0
         iwa3(ic) = iwa1(0)
         if (iwa1(0) .gt. 0) iwa2(iwa1(0)) = ic
         iwa1(0) = ic
         iwa4(jp) = 0
         list(jp) = 0
   10    continue
c
c     Determine the maximal search length for the list
c     of columns of maximal incidence.
c
      maxlst = 0
      do 20 ir = 1, m
         maxlst = maxlst + (ipntr(ir+1) - ipntr(ir))**2
   20    continue
      maxlst = maxlst/n
      maxclq = 0
      numord = 1
c
c     Beginning of iteration loop.
c
   30 continue
c
c        Choose a column jcol of maximal degree among the
c        columns of maximal incidence maxinc.
c
   40    continue
            jp = iwa1(maxinc)
            if (jp .gt. 0) go to 50
            maxinc = maxinc - 1
            go to 40
   50    continue
         numwgt = -1
         do 60 numlst = 1, maxlst
            if (ndeg(jp) .gt. numwgt) then
               numwgt = ndeg(jp)
               jcol = jp
               end if
            jp = iwa3(jp)
            if (jp .le. 0) go to 70
   60       continue
   70    continue
         list(jcol) = numord
c
c        Update the size of the largest clique
c        found during the ordering.
c
         if (maxinc .eq. 0) ncomp = 0
         ncomp = ncomp + 1
         if (maxinc+1 .eq. ncomp) maxclq = max(maxclq,ncomp)
c
c        Termination test.
c
         numord = numord + 1
         if (numord .gt. n) go to 100
c
c        Delete column jcol from the maxinc list.
c
         if (iwa2(jcol) .eq. 0) then
            iwa1(maxinc) = iwa3(jcol)
         else
            iwa3(iwa2(jcol)) = iwa3(jcol)
            end if
         if (iwa3(jcol) .gt. 0) iwa2(iwa3(jcol)) = iwa2(jcol)
c
c        Find all columns adjacent to column jcol.
c
         iwa4(jcol) = n
c
c        Determine all positions (ir,jcol) which correspond
c        to non-zeroes in the matrix.
c
         do 90 jp = jpntr(jcol), jpntr(jcol+1)-1
            ir = indrow(jp)
c
c           For each row ir, determine all positions (ir,ic)
c           which correspond to non-zeroes in the matrix.
c
            do 80 ip = ipntr(ir), ipntr(ir+1)-1
               ic = indcol(ip)
c
c              Array iwa4 marks columns which are adjacent to
c              column jcol.
c
               if (iwa4(ic) .lt. numord) then
                  iwa4(ic) = numord
c
c                 Update the pointers to the current incidence lists.
c
                  numinc = list(ic)
                  list(ic) = list(ic) + 1
                  maxinc = max(maxinc,list(ic))
c
c                 Delete column ic from the numinc list.
c
                  if (iwa2(ic) .eq. 0) then
                     iwa1(numinc) = iwa3(ic)
                  else
                     iwa3(iwa2(ic)) = iwa3(ic)
                     end if
                  if (iwa3(ic) .gt. 0) iwa2(iwa3(ic)) = iwa2(ic)
c
c                 Add column ic to the numinc+1 list.
c
                  iwa2(ic) = 0
                  iwa3(ic) = iwa1(numinc+1)
                  if (iwa1(numinc+1) .gt. 0) iwa2(iwa1(numinc+1)) = ic
                  iwa1(numinc+1) = ic
                  end if
   80          continue
   90       continue
c
c        End of iteration loop.
c
         go to 30
  100 continue
c
c     Invert the array list.
c
      do 110 jcol = 1, n
         iwa2(list(jcol)) = jcol
  110    continue
      do 120 jp = 1, n
         list(jp) = iwa2(jp)
  120    continue
      return
c
c     Last card of subroutine ido.
c
      end

      subroutine idog(n,nghbrp,npntrp,nghbrs,npntrs,listp,
     *                maxclq,maxid,iwa1,iwa2,iwa3)
      integer n,maxclq,maxid
      integer nghbrp(*),npntrp(n+1),nghbrs(*),npntrs(n+1),listp(n),
     *        iwa1(0:n-1),iwa2(n),iwa3(n)
c     **********
c
c     subroutine idog
c
c     Given a loopless graph G = (V,E), this subroutine determines
c     the incidence degree ordering of the vertices of G.
c
c     The incidence degree ordering is determined recursively by
c     letting list(k), k = 1,...,n be a vertex with maximal
c     incidence to the subgraph spanned by the ordered vertices.
c     Among all the vertices of maximal incidence, a vertex of
c     maximal degree is chosen. This subroutine determines the
c     inverse of the incidence degree ordering, that is, an array
c     listp such that listp(list(k)) = k for k = 1,2,...,n.
c
c     The subroutine statement is
c
c       subroutine idog(n,nghbrp,npntrp,nghbrs,npntrs,listp,
c                       maxclq,maxid,iwa1,iwa2,iwa3)
c
c     where
c
c       n is a positive integer input variable set to the number
c         of vertices of G.
c
c       nghbrp is an integer input array which contains the
c         predecessor adjacency lists for the graph G.
c
c       npntrp is an integer input array of length n + 1 which
c         specifies the locations of the predecessor adjacency
c         lists in nghbrp. The vertices preceding and adjacent
c         to vertex j are
c
c               nghbrp(k), k = npntrp(j),...,npntrp(j+1)-1.
c
c         Note that npntrp(n+1)-1 is then the number of vertices
c         plus edges of the graph G.
c
c       nghbrs is an integer input array which contains the
c         successor adjacency lists for the graph G.
c
c       npntrs is an integer input array of length n + 1 which
c         specifies the locations of the successor adjacency
c         lists in nghbrs. The vertices succeeding and adjacent
c         to vertex j are
c
c               nghbrs(k), k = npntrs(j),...,npntrs(j+1)-1.
c
c         Note that npntrs(n+1)-1 is then the number of vertices
c         plus edges of the graph G.
c
c       listp is an integer output array of length n which specifies
c         the inverse of the incidence degree ordering of the
c         vertices. Vertex j is in position listp(j) of this ordering.
c
c       maxclq is an integer output variable set to the size
c         of the largest clique found during the ordering.
c
c       maxid is an integer output variable set to the maximum
c         incidence degree found during the ordering.
c
c       iwa1,iwa2, and iwa3 are integer work arrays of length n.
c
c     Subprograms called
c
c       MINPACK-supplied ... numsrt
c
c       FORTRAN-supplied ... max
c
c     Argonne National Laboratory. MINPACK Project. December 1984.
c     Thomas F. Coleman, Burton S. Garbow, Jorge J. More'
c
c     **********
      integer i,j,k,maxinc,maxdeg,maxlst,ncomp,numdeg,numinc,numord
c
c     Initialization block.
c
      do 10 j = 1, n
         listp(j) = (npntrp(j+1) - npntrp(j) - 1) +
     *              (npntrs(j+1) - npntrs(j) - 1)
   10    continue
      maxlst = (npntrp(n+1) + npntrs(n+1))/n
c
c     Sort the degree sequence.
c
      call numsrt(n,n-1,listp,1,iwa1,iwa2,iwa3)
c
c     Create a doubly-linked list to access the incidences of the
c     vertices. The pointers for the linked list are as follows.
c
c     Each un-ordered vertex i is in a list (the incidence list)
c     of vertices with the same incidence.
c
c     iwa1(numinc) is the first vertex in the numinc list
c     unless iwa1(numinc) = 0. In this case there are
c     no vertices in the numinc list.
c
c     iwa2(i) is the vertex before i in the incidence list
c     unless iwa2(i) = 0. In this case i is the first
c     vertex in this incidence list.
c
c     iwa3(i) is the vertex after i in the incidence list
c     unless iwa3(i) = 0. In this case i is the last
c     vertex in this incidence list.
c
c     If i is an un-ordered vertex, then -listp(i) is the
c     incidence of i to the graph induced by the ordered
c     vertices. If j is an ordered vertex, then listp(j)
c     is the incidence degree order of vertex j.
c
      maxinc = 0
      do 20 j = 1, n
         i = iwa1(j-1)
         iwa1(j-1) = 0
         iwa2(i) = 0
         iwa3(i) = iwa1(0)
         if (iwa1(0) .gt. 0) iwa2(iwa1(0)) = i
         iwa1(0) = i
         listp(j) = 0
   20    continue
      maxclq = 0
      maxid = 0
      numord = 1
c
c     Beginning of iteration loop.
c
   30 continue
c
c        Choose a vertex j of maximal degree among the
c        vertices of maximal incidence maxinc.
c
   40    continue
            k = iwa1(maxinc)
            if (k .gt. 0) go to 50
            maxinc = maxinc - 1
            go to 40
   50    continue
         maxdeg = -1
         do 60 i = 1, maxlst
            numdeg = (npntrp(k+1) - npntrp(k) - 1) +
     *               (npntrs(k+1) - npntrs(k) - 1)
            if (numdeg .gt. maxdeg) then
               maxdeg = numdeg
               j = k
               end if
            k = iwa3(k)
            if (k .le. 0) go to 70
   60       continue
   70    continue
         listp(j) = numord
         maxid = max(maxid,maxinc)
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
c        Delete vertex j from the maxinc list.
c
         if (iwa2(j) .eq. 0) then
            iwa1(maxinc) = iwa3(j)
         else
            iwa3(iwa2(j)) = iwa3(j)
            end if
         if (iwa3(j) .gt. 0) iwa2(iwa3(j)) = iwa2(j)
c
c        Determine all the neighbors of vertex j which precede j
c        in the subgraph spanned by the un-ordered vertices.
c
         do 80 k = npntrp(j), npntrp(j+1)-1
            i = nghbrp(k)
c
c           Update the pointers to the current incidence lists.
c
            numinc = -listp(i)
            if (numinc .ge. 0) then
               listp(i) = listp(i) - 1
               maxinc = max(maxinc,-listp(i))
c
c              Delete vertex i from the numinc list.
c
               if (iwa2(i) .eq. 0) then
                  iwa1(numinc) = iwa3(i)
               else
                  iwa3(iwa2(i)) = iwa3(i)
                  end if
               if (iwa3(i) .gt. 0) iwa2(iwa3(i)) = iwa2(i)
c
c              Add vertex i to the numinc+1 list.
c
               iwa2(i) = 0
               iwa3(i) = iwa1(numinc+1)
               if (iwa1(numinc+1) .gt. 0) iwa2(iwa1(numinc+1)) = i
               iwa1(numinc+1) = i
               end if
   80       continue
c
c        Determine all the neighbors of vertex j which succeed j
c        in the subgraph spanned by the un-ordered vertices.
c
         do 90 k = npntrs(j), npntrs(j+1)-1
            i = nghbrs(k)
c
c           Update the pointers to the current incidence lists.
c
            numinc = -listp(i)
            if (numinc .ge. 0) then
               listp(i) = listp(i) - 1
               maxinc = max(maxinc,-listp(i))
c
c              Delete vertex i from the numinc list.
c
               if (iwa2(i) .eq. 0) then
                  iwa1(numinc) = iwa3(i)
               else
                  iwa3(iwa2(i)) = iwa3(i)
                  end if
               if (iwa3(i) .gt. 0) iwa2(iwa3(i)) = iwa2(i)
c
c              Add vertex i to the numinc+1 list.
c
               iwa2(i) = 0
               iwa3(i) = iwa1(numinc+1)
               if (iwa1(numinc+1) .gt. 0) iwa2(iwa1(numinc+1)) = i
               iwa1(numinc+1) = i
               end if
   90       continue
c
c        End of iteration loop.
c
         go to 30
  100 continue
      return
c
c     Last card of subroutine idog.
c
      end

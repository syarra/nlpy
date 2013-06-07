      subroutine slog(n,nghbrp,npntrp,nghbrs,npntrs,listp,
     *                maxclq,maxvd,iwa1,iwa2,iwa3)
      integer n,maxclq,maxvd
      integer nghbrp(*),npntrp(n+1),nghbrs(*),npntrs(n+1),listp(n),
     *        iwa1(0:n-1),iwa2(n),iwa3(n)
c     **********
c
c     subroutine slog
c
c     Given a loopless graph G = (V,E), this subroutine determines
c     the smallest-last ordering of the vertices of G.
c
c     The smallest-last ordering is determined recursively by
c     letting list(k), k = n,...,1 be a vertex with least degree
c     in the subgraph spanned by the un-ordered vertices.
c     This subroutine determines the inverse of the smallest-last
c     ordering, that is, an array listp such that listp(list(k)) = k
c     for k = 1,2,...,n.
c
c     The subroutine statement is
c
c       subroutine slog(n,nghbrp,npntrp,nghbrs,npntrs,listp,
c                       maxclq,maxvd,iwa1,iwa2,iwa3)
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
c         the inverse of the smallest-last ordering of the vertices.
c         Vertex j is in position listp(j) of this ordering.
c
c       maxclq is an integer output variable set to the size
c         of the largest clique found during the ordering.
c
c       maxvd is an integer output variable set to the maximum
c         vertex degree found during the ordering.
c
c       iwa1,iwa2, and iwa3 are integer work arrays of length n.
c
c     Subprograms called
c
c       FORTRAN-supplied ... max,min
c
c     Argonne National Laboratory. MINPACK Project. December 1984.
c     Thomas F. Coleman, Burton S. Garbow, Jorge J. More'
c
c     **********
      integer i,j,k,mindeg,numdeg,numord
c
c     Initialization block.
c
      mindeg = n
      do 10 j = 1, n
         iwa1(j-1) = 0
         listp(j) = (npntrp(j) - npntrp(j+1) + 1) +
     *              (npntrs(j) - npntrs(j+1) + 1)
         mindeg = min(mindeg,-listp(j))
   10    continue
c
c     Create a doubly-linked list to access the degrees of the
c     vertices. The pointers for the linked list are as follows.
c
c     Each un-ordered vertex i is in a list (the degree list)
c     of vertices with the same degree.
c
c     iwa1(numdeg) is the first vertex in the numdeg list
c     unless iwa1(numdeg) = 0. In this case there are
c     no vertices in the numdeg list.
c
c     iwa2(i) is the vertex before i in the degree list
c     unless iwa2(i) = 0. In this case i is the first
c     vertex in this degree list.
c
c     iwa3(i) is the vertex after i in the degree list
c     unless iwa3(i) = 0. In this case i is the last
c     vertex in this degree list.
c
c     If i is an un-ordered vertex, then -listp(i) is the
c     degree of i in the graph induced by the un-ordered
c     vertices. If j is an ordered vertex, then listp(j)
c     is the smallest-last order of vertex j.
c
      do 20 j = 1, n
         numdeg = -listp(j)
         iwa2(j) = 0
         iwa3(j) = iwa1(numdeg)
         if (iwa1(numdeg) .gt. 0) iwa2(iwa1(numdeg)) = j
         iwa1(numdeg) = j
   20    continue
      maxclq = 0
      maxvd = 0
      numord = n
c
c     Beginning of iteration loop.
c
   30 continue
c
c        Choose a vertex j of minimal degree mindeg.
c
   40    continue
            j = iwa1(mindeg)
            if (j .gt. 0) go to 50
            mindeg = mindeg + 1
            go to 40
   50    continue
         listp(j) = numord
         maxvd = max(maxvd,mindeg)
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
c        Delete vertex j from the mindeg list.
c
         iwa1(mindeg) = iwa3(j)
         if (iwa3(j) .gt. 0) iwa2(iwa3(j)) = 0
c
c        Determine all the neighbors of vertex j which precede j
c        in the subgraph spanned by the un-ordered vertices.
c
         do 60 k = npntrp(j), npntrp(j+1)-1
            i = nghbrp(k)
c
c           Update the pointers to the current degree lists.
c
            numdeg = -listp(i)
            if (numdeg .ge. 0) then
               listp(i) = listp(i) + 1
               mindeg = min(mindeg,-listp(i))
c
c              Delete vertex i from the numdeg list.
c
               if (iwa2(i) .eq. 0) then
                  iwa1(numdeg) = iwa3(i)
               else
                  iwa3(iwa2(i)) = iwa3(i)
                  end if
               if (iwa3(i) .gt. 0) iwa2(iwa3(i)) = iwa2(i)
c
c              Add vertex i to the numdeg-1 list.
c
               iwa2(i) = 0
               iwa3(i) = iwa1(numdeg-1)
               if (iwa1(numdeg-1) .gt. 0) iwa2(iwa1(numdeg-1)) = i
               iwa1(numdeg-1) = i
               end if
   60       continue
c
c        Determine all the neighbors of vertex j which succeed j
c        in the subgraph spanned by the un-ordered vertices.
c
         do 70 k = npntrs(j), npntrs(j+1)-1
            i = nghbrs(k)
c
c           Update the pointers to the current degree lists.
c
            numdeg = -listp(i)
            if (numdeg .ge. 0) then
               listp(i) = listp(i) + 1
               mindeg = min(mindeg,-listp(i))
c
c              Delete vertex i from the numdeg list.
c
               if (iwa2(i) .eq. 0) then
                  iwa1(numdeg) = iwa3(i)
               else
                  iwa3(iwa2(i)) = iwa3(i)
                  end if
               if (iwa3(i) .gt. 0) iwa2(iwa3(i)) = iwa2(i)
c
c              Add vertex i to the numdeg-1 list.
c
               iwa2(i) = 0
               iwa3(i) = iwa1(numdeg-1)
               if (iwa1(numdeg-1) .gt. 0) iwa2(iwa1(numdeg-1)) = i
               iwa1(numdeg-1) = i
               end if
   70       continue
c
c        End of iteration loop.
c
         go to 30
   80 continue
      return
c
c     Last card of subroutine slog.
c
      end

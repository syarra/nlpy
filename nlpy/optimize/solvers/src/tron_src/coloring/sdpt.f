      subroutine sdpt(n,nghbrp,npntrp,nghbrs,npntrs,ngrp,maxgrp,
     *                iwa1,iwa2)
      integer n,maxgrp
      integer nghbrp(*),npntrp(n+1),nghbrs(*),npntrs(n+1),ngrp(n),
     *        iwa1(0:n-1),iwa2(n)
c     **********
c
c     subroutine sdpt
c
c     Given a loopless graph G = (V,E), this subroutine determines
c     a symmetric coloring of G by the Powell-Toint direct method.
c
c     The Powell-Toint method assigns the k-th color by examining
c     the un-colored vertices U(k) in order of non-increasing degree
c     and assigning color k to vertex v if there are no paths of
c     length 1 or 2 (in the graph induced by U(k)) between v and
c     some k-colored vertex.
c
c     The subroutine statement is
c
c       subroutine sdpt(n,nghbrp,npntrp,nghbrs,npntrs,ngrp,maxgrp,
c                       iwa1,iwa2)
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
c       ngrp is an integer output array of length n which specifies
c         the symmetric coloring of G. Vertex j is colored with
c         color ngrp(j).
c
c       maxgrp is an integer output variable which specifies the
c         number of colors in the symmetric coloring of G.
c
c       iwa1 and iwa2 are integer work arrays of length n.
c
c     Subprograms called
c
c       FORTRAN-supplied ... max
c
c     Argonne National Laboratory. MINPACK Project. December 1984.
c     Thomas F. Coleman, Burton S. Garbow, Jorge J. More'
c
c     **********
      integer j,jp,k,kp,l,maxdeg,numdeg,numv
c
c     Initialization block. Numv is the current number of un-colored
c     vertices, maxdeg is the maximum induced degree of these
c     vertices, and maxgrp is the current group number (color).
c
      numv = n
      maxdeg = 0
      do 10 j = 1, n
         ngrp(j) = (npntrp(j) - npntrp(j+1) + 1) +
     *             (npntrs(j) - npntrs(j+1) + 1)
         maxdeg = max(maxdeg,-ngrp(j))
         iwa2(j) = -j
   10    continue
      maxgrp = 0
c
c     Beginning of iteration loop.
c
   20 continue
c
c        Sort the list of un-colored vertices so that their
c        induced degrees are in non-decreasing order.
c
         do 30 numdeg = 0, maxdeg
            iwa1(numdeg) = 0
   30       continue
         do 40 l = 1, numv
            numdeg = -ngrp(-iwa2(l))
            iwa1(numdeg) = iwa1(numdeg) + 1
   40       continue
         k = 1
         do 50 numdeg = maxdeg, 0, -1
            l = iwa1(numdeg)
            iwa1(numdeg) = k
            k = k + l
   50       continue
         k = 1
   60    continue
            j = iwa2(k)
            if (j .gt. 0) then
               k = iwa1(-ngrp(j))
            else
               numdeg = -ngrp(-j)
               l = iwa1(numdeg)
               iwa2(k) = iwa2(l)
               iwa2(l) = -j
               iwa1(numdeg) = iwa1(numdeg) + 1
               end if
            if (k .le. numv) go to 60
         maxgrp = maxgrp + 1
c
c        Determine the vertices in group maxgrp.
c
         do 160 l = 1, numv
            j = iwa2(l)
c
c           Examine each vertex k preceding vertex j and all
c           the neighbors of vertex k to determine if vertex
c           j can be considered for group maxgrp.
c
            do 90 jp = npntrp(j), npntrp(j+1)-1
               k = nghbrp(jp)
               if (ngrp(k) .eq. maxgrp) go to 150
               if (ngrp(k) .le. 0) then
                  do 70 kp = npntrp(k), npntrp(k+1)-1
                     if (ngrp(nghbrp(kp)) .eq. maxgrp) go to 150
   70                continue
                  do 80 kp = npntrs(k), npntrs(k+1)-1
                     if (ngrp(nghbrs(kp)) .eq. maxgrp) go to 150
   80                continue
                  end if
   90          continue
c
c           Examine each vertex k succeeding vertex j and all
c           the neighbors of vertex k to determine if vertex
c           j can be added to group maxgrp.
c
            do 120 jp = npntrs(j), npntrs(j+1)-1
               k = nghbrs(jp)
               if (ngrp(k) .eq. maxgrp) go to 150
               if (ngrp(k) .le. 0) then
                  do 100 kp = npntrp(k), npntrp(k+1)-1
                     if (ngrp(nghbrp(kp)) .eq. maxgrp) go to 150
  100                continue
                  do 110 kp = npntrs(k), npntrs(k+1)-1
                     if (ngrp(nghbrs(kp)) .eq. maxgrp) go to 150
  110                continue
                  end if
  120          continue
c
c           Add vertex j to group maxgrp and remove vertex j
c           from the list of un-colored vertices.
c
            ngrp(j) = maxgrp
            iwa2(l) = 0
c
c           Update the degrees of the neighbors of vertex j.
c
            do 130 jp = npntrp(j), npntrp(j+1)-1
               k = nghbrp(jp)
               if (ngrp(k) .lt. 0) ngrp(k) = ngrp(k) + 1
  130          continue
            do 140 jp = npntrs(j), npntrs(j+1)-1
               k = nghbrs(jp)
               if (ngrp(k) .lt. 0) ngrp(k) = ngrp(k) + 1
  140          continue
  150       continue
  160       continue
c
c        Compress the updated list of un-colored vertices.
c        Reset numv and recompute maxdeg.
c
         k = 0
         maxdeg = 0
         do 170 l = 1, numv
            if (iwa2(l) .ne. 0) then
               k = k + 1
               iwa2(k) = -iwa2(l)
               maxdeg = max(maxdeg,-ngrp(iwa2(l)))
               end if
  170       continue
         numv = k
c
c        End of iteration loop.
c
         if (numv .gt. 0) go to 20
      return
c
c     Last card of subroutine sdpt.
c
      end

      subroutine dssm(n,npairs,indrow,indcol,method,listp,ngrp,
     *                maxgrp,mingrp,info,ipntr,jpntr,iwa,liwa)
      integer n,npairs,method,maxgrp,mingrp,info,liwa
      integer indrow(npairs),indcol(npairs),listp(n),ngrp(n),
     *        ipntr(n+1),jpntr(n+1),iwa(liwa)
c     **********
c
c     subroutine dssm
c
c     Given the sparsity pattern of a symmetric matrix A of order n,
c     this subroutine determines a symmetric permutation of A and a
c     partition of the columns of A consistent with the determination
c     of A by a lower triangular substitution method.
c
c     The sparsity pattern of the matrix A is specified by the
c     arrays indrow and indcol. On input the indices for the
c     non-zero elements in the lower triangular part of A are
c
c           (indrow(k),indcol(k)), k = 1,2,...,npairs.
c
c     The (indrow(k),indcol(k)) pairs may be specified in any order.
c     Duplicate input pairs are permitted, but the subroutine
c     eliminates them. The subroutine requires that all the diagonal
c     elements be part of the sparsity pattern and replaces any pair
c     (indrow(k),indcol(k)) where indrow(k) is less than indcol(k)
c     by the pair (indcol(k),indrow(k)).
c
c     The direct method (method = 1) first determines a partition
c     of the columns of A such that two columns in a group have a
c     non-zero element in row k only if column k is in an earlier
c     group. Using this partition, the subroutine then computes a
c     symmetric permutation of A consistent with the determination
c     of A by a lower triangular substitution method.
c
c     The indirect method first computes a symmetric permutation of A
c     which minimizes the maximum number of non-zero elements in any
c     row of L, where L is the lower triangular part of the permuted
c     matrix. The subroutine then partitions the columns of L into
c     groups such that columns of L in a group do not have a non-zero
c     in the same row position.
c
c     The subroutine statement is
c
c       subroutine dssm(n,npairs,indrow,indcol,method,listp,ngrp,
c                       maxgrp,mingrp,info,ipntr,jpntr,iwa,liwa)
c
c     where
c
c       n is a positive integer input variable set to the order of A.
c
c       npairs is a positive integer input variable set to the number
c         of (indrow,indcol) pairs used to describe the sparsity
c         pattern of A.
c
c       indrow is an integer array of length npairs. On input indrow
c         must contain the row indices of the non-zero elements in
c         the lower triangular part of A. On output indrow is
c         permuted so that the corresponding column indices are in
c         non-decreasing order. The column indices can be recovered
c         from the array jpntr.
c
c       indcol is an integer array of length npairs. On input indcol
c         must contain the column indices of the non-zero elements
c         in the lower triangular part of A. On output indcol is
c         permuted so that the corresponding row indices are in
c         non-decreasing order. The row indices can be recovered
c         from the array ipntr.
c
c       method is an integer input variable. If method = 1, the
c         direct method is used to determine the partition and
c         symmetric permutation. Otherwise, the indirect method is
c         used to determine the symmetric permutation and partition.
c
c       listp is an integer output array of length n which specifies
c         the symmetric permutation of the matrix A. Element (i,j)
c         of A is the (listp(i),listp(j)) element of the permuted
c         matrix.
c
c       ngrp is an integer output array of length n which specifies
c         the partition of the columns of A. Column j belongs to
c         group ngrp(j).
c
c       maxgrp is an integer output variable which specifies the
c         number of groups in the partition of the columns of A.
c
c       mingrp is an integer output variable which specifies a lower
c         bound for the number of groups in any partition of the
c         columns of A consistent with the determination of A by a
c         lower triangular substitution method.
c
c       info is an integer output variable set as follows. For
c         normal termination info = 1. If n or npairs is not
c         positive or liwa is less than 6*n, then info = 0. If the
c         k-th element of indrow or the k-th element of indcol is
c         not an integer between 1 and n, or if the k-th diagonal
c         element is not in the sparsity pattern, then info = -k.
c
c       ipntr is an integer output array of length n + 1 which
c         specifies the locations of the column indices in indcol.
c         The column indices for row i are
c
c               indcol(k), k = ipntr(i),...,ipntr(i+1)-1.
c
c         Note that ipntr(n+1)-1 is then the number of non-zero
c         elements in the lower triangular part of the matrix A.
c
c       jpntr is an integer output array of length n + 1 which
c         specifies the locations of the row indices in indrow.
c         The row indices for column j are
c
c               indrow(k), k = jpntr(j),...,jpntr(j+1)-1.
c
c         Note that jpntr(n+1)-1 is then the number of non-zero
c         elements in the lower triangular part of the matrix A.
c
c       iwa is an integer work array of length liwa.
c
c       liwa is a positive integer input variable not less than 6*n.
c
c     Subprograms called
c
c       MINPACK-supplied ... degr,ido,idog,numsrt,sdpt,seq,setr,
c                            slo,slog,srtdat
c
c       FORTRAN-supplied ... max,min
c
c     Argonne National Laboratory. MINPACK Project. December 1984.
c     Thomas F. Coleman, Burton S. Garbow, Jorge J. More'
c
c     **********
      integer i,ir,j,jp,k,maxid,maxvd,maxclq,nnz,numgrp
c
c     Check the input data.
c
      info = 0
      if (n .lt. 1 .or. npairs .lt. 1 .or. liwa .lt. 6*n) return
      do 10 k = 1, n
         iwa(k) = 0
   10    continue
      do 20 k = 1, npairs
         info = -k
         if (indrow(k) .lt. 1 .or. indrow(k) .gt. n .or.
     *       indcol(k) .lt. 1 .or. indcol(k) .gt. n) return
         if (indrow(k) .eq. indcol(k)) iwa(indrow(k)) = 1
   20    continue
      do 30 k = 1, n
         info = -k
         if (iwa(k) .ne. 1) return
   30    continue
      info = 1
c
c     Generate the sparsity pattern for the lower
c     triangular part of A.
c
      do 40 k = 1, npairs
         i = indrow(k)
         j = indcol(k)
         indrow(k) = max(i,j)
         indcol(k) = min(i,j)
   40    continue
c
c     Sort the data structure by columns.
c
      call srtdat(n,npairs,indrow,indcol,jpntr,iwa)
c
c     Compress the data and determine the number of non-zero
c     elements in the lower triangular part of A.
c
      do 50 i = 1, n
         iwa(i) = 0
   50    continue
      nnz = 0
      do 70 j = 1, n
         k = nnz
         do 60 jp = jpntr(j), jpntr(j+1)-1
            ir = indrow(jp)
            if (iwa(ir) .ne. j) then
               nnz = nnz + 1
               indrow(nnz) = ir
               iwa(ir) = j
               end if
   60       continue
         jpntr(j) = k + 1
   70    continue
      jpntr(n+1) = nnz + 1
c
c     Extend the data structure to rows.
c
      call setr(n,n,indrow,jpntr,indcol,ipntr,iwa)
c
c     Determine the smallest-last ordering of the vertices of the
c     adjacency graph of A, and from it determine a lower bound
c     for the number of groups.
c
      call slog(n,indrow,jpntr,indcol,ipntr,iwa(1),maxclq,
     *          maxvd,iwa(n+1),iwa(2*n+1),iwa(3*n+1))
      mingrp =  1 + maxvd
c
c     Use the selected method.
c
      if (method .eq. 1) then
c
c        Direct method. Determine a partition of the columns
c        of A by the Powell-Toint method.
c
         call sdpt(n,indrow,jpntr,indcol,ipntr,ngrp,maxgrp,
     *             iwa(n+1),iwa(2*n+1))
c
c        Define a symmetric permutation of A according to the
c        ordering of the column group numbers in the partition.
c
         call numsrt(n,maxgrp,ngrp,1,iwa(1),iwa(2*n+1),iwa(n+1))
         do 80 i = 1, n
            listp(iwa(i)) = i
   80       continue
      else
c
c        Indirect method. Determine the incidence degree ordering
c        of the vertices of the adjacency graph of A and, together
c        with the smallest-last ordering, define a symmetric
c        permutation of A.
c
         call idog(n,indrow,jpntr,indcol,ipntr,listp,maxclq,
     *             maxid,iwa(n+1),iwa(2*n+1),iwa(3*n+1))
         if (maxid .gt. maxvd) then
            do 90 i = 1, n
               listp(i) = iwa(i)
   90          continue
            end if
c
c        Generate the sparsity pattern for the lower
c        triangular part L of the permuted matrix.
c
         do 110 j = 1, n
            do 100 jp = jpntr(j), jpntr(j+1)-1
               i = indrow(jp)
               indrow(jp) = max(listp(i),listp(j))
               indcol(jp) = min(listp(i),listp(j))
  100          continue
  110       continue
c
c        Sort the data structure by columns.
c
         call srtdat(n,nnz,indrow,indcol,jpntr,iwa)
c
c        Extend the data structure to rows.
c
         call setr(n,n,indrow,jpntr,indcol,ipntr,iwa)
c
c        Determine the degree sequence for the intersection
c        graph of the columns of L.
c
         call degr(n,indrow,jpntr,indcol,ipntr,iwa(5*n+1),iwa(n+1))
c
c        Color the intersection graph of the columns of L
c        with the smallest-last (SL) ordering.
c
         call slo(n,indrow,jpntr,indcol,ipntr,iwa(5*n+1),iwa(4*n+1),
     *            maxclq,iwa(1),iwa(n+1),iwa(2*n+1),iwa(3*n+1))
         call seq(n,indrow,jpntr,indcol,ipntr,iwa(4*n+1),iwa(1),
     *            maxgrp,iwa(n+1))
         do 120 j = 1, n
            ngrp(j) = iwa(listp(j))
  120       continue
c
c        Exit if the smallest-last ordering is optimal.
c
         if (maxgrp .eq. maxclq) go to 140
c
c        Color the intersection graph of the columns of L
c        with the incidence degree (ID) ordering.
c
         call ido(n,n,indrow,jpntr,indcol,ipntr,iwa(5*n+1),iwa(4*n+1),
     *            maxclq,iwa(1),iwa(n+1),iwa(2*n+1),iwa(3*n+1))
         call seq(n,indrow,jpntr,indcol,ipntr,iwa(4*n+1),iwa(1),
     *            numgrp,iwa(n+1))
c
c        Retain the better of the two orderings.
c
         if (numgrp .lt. maxgrp) then
            maxgrp = numgrp
            do 130 j = 1, n
               ngrp(j) = iwa(listp(j))
  130          continue
            end if
  140    continue
c
c        Generate the sparsity pattern for the lower
c        triangular part of the original matrix.
c
         do 150 j = 1, n
            iwa(listp(j)) = j
  150       continue
         do 170 j = 1, n
            do 160 jp = jpntr(j), jpntr(j+1)-1
               i = indrow(jp)
               indrow(jp) = max(iwa(i),iwa(j))
               indcol(jp) = min(iwa(i),iwa(j))
  160          continue
  170       continue
c
c        Sort the data structure by columns.
c
         call srtdat(n,nnz,indrow,indcol,jpntr,iwa)
c
c        Extend the data structure to rows.
c
         call setr(n,n,indrow,jpntr,indcol,ipntr,iwa)
         end if
      return
c
c     Last card of subroutine dssm.
c
      end

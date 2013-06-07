      subroutine dsphesd(n,row_ind,col_ind,row_ptr,col_ptr,listp,ngrp,
     +                  maxgrp,numgrp,eta,fhesd,fhes,diag,iwa)
      integer n, maxgrp, numgrp
      integer row_ind(*), col_ptr(n+1), col_ind(*), row_ptr(n+1)
      integer listp(n), ngrp(n)
      integer iwa(n)
      double precision eta(n), fhesd(n), fhes(*), diag(n)
c     **********
c
c     Subroutine dsphesd
c
c     This subroutine computes an approximation to the (symmetric)
c     Hessian matrix of a function by a substitution method.
c     The lower triangular part of the approximation is stored
c     in compressed column storage.
c
c     This subroutine requires a symmetric permutation of the
c     Hessian matrix and a partition of the columns of the Hessian
c     matrix consistent with the determination of the Hessian
c     matrix by a lower triangular substitution method.
c     This information can be provided by subroutine dssm.
c
c     The symmetric permutation of the Hessian matrix is defined
c     by the array listp. This array is only used internally.
c
c     The partition of the Hessian matrix is defined by the array
c     ngrp by setting ngrp(j) to the group number of column j.
c     The user must provide an approximation to the columns of
c     the Hessian matrix in each group by specifying a difference
c     parameter vector eta and an approximation to A*d where A is
c     the Hessian matrix and the vector d is defined by the
c     following section of code.
c
c           do j = 1, n
c              d(j) = 0.0
c              if (ngrp(j) .eq. numgrp) d(j) = eta(j)
c           end do
c
c     In the above code numgrp is a group number and eta(j) is the
c     difference parameter used to approximate column j of the
c     Hessian matrix. Suitable values for eta(j) must be provided.
c
c     As mentioned above, an approximation to A*d must be provided.
c     For example, if grad f(x) is the gradient of the function at x,
c     then
c
c           grad f(x+d) - grad f(x)
c
c     corresponds to the forward difference approximation.
c
c     The lower triangular substitution method requires that the
c     approximations to A*d for all the groups be stored in special
c     locations of the array fhes. This is done by calls with
c     numgrp = 1, 2, ... ,maxgrp. On the call with numgrp = maxgrp, 
c     the array fhes is overwritten with the approximation to the 
c     lower triangular part of the Hessian matrix.
c
c     The subroutine statement is
c
c       subroutine dsphesd(n,row_ind,col_ind,row_ptr,col_ptr,listp,ngrp,
c                         maxgrp,numgrp,eta,fhesd,fhes,diag,iwa)
c
c     where
c
c       n is an integer variable.
c         On entry n is the number of variables.
c         On exit n is unchanged.
c
c       row_ind is an integer array of dimension nnz.
c         On entry row_ind must contain row indices for the strict 
c            lower triangular part of A in compressed column storage.
c         On exit row_ind is unchanged.
c
c       col_ind is an integer array of dimension nnz.
c         On entry col_ind must contain column indices for the strict 
c            lower triangular part of A in compressed column storage.
c         On exit col_ind is unchanged.
c
c       row_ptr is an integer array of dimension n + 1.
c         On entry row_ptr must contain pointers to the rows of A.
c            The nonzeros in row j of A must be in positions
c            row_ptr(j), ... , row_ptr(j+1) - 1 of col_ind.
c         On exit row_ptr is unchanged.
c
c       col_ptr is an integer array of dimension n + 1.
c         On entry col_ptr must contain pointers to the columns of A.
c            The nonzeros in column j of A must be in positions
c            col_ptr(j), ... , col_ptr(j+1) - 1 of row_ind.
c         On exit col_ptr is unchanged.
c
c       listp is an integer array of length n.
c         On input listp need not be specified.
c         On output listp specifies a permutation of the matrix. 
c            Element (i,j) of the matrix is the (listp(i),listp(j))
c            element of the permuted matrix.
c
c       ngrp is an integer array of length n.
c         On entry ngrp specifies the partition of the columns
c            of A. Column j belongs to group ngrp(j).
c         On exit ngrp is unchanged.
c
c       maxgrp is an integer variable. 
c         On entry maxgrp specifies the number of groups in
c            the partition of the columns of A.
c         On exit maxgrp is unchanged.
c
c       numgrp is an integer variable.
c         On input numgrp must be set to a group number.
c         On output numgrp is unchanged.
c
c       eta is a double precision variable.
c         On input eta is the difference parameter vector.
c         On output eta is unchanged.
c
c       fhesd is a double precision array of length n.
c         On input fhesd contains an approximation to A*d, where A 
c           is the Hessian matrix and d is the difference vector for 
c           group numgrp.
c         On output fhesd is unchanged.
c
c       fhes is a double precision array of length nnz.
c         On input fhes need not be specified.
c         On output fhes is overwritten. When numgrp = maxgrp the
c            array fhes contains an approximation to the Hessian matrix
c            in compressed column storage. The elements in column j of
c            the strict lower triangular part of the Hessian matrix are
c
c               fhes(k), k = col_ptr(j),...,col_ptr(j+1)-1,
c
c            and the row indices for these elements are
c
c               row_ind(k), k = col_ptr(j),...,col_ptr(j+1)-1.
c
c       diag is a double precision array of length n.
c         On input diag need not be specified.
c         On output diag is overwritten. When numgrp = maxgrp the array
c            diag contains the diagonal elements of an approximation to 
c            the Hessian matrix.
c
c       iwa is an integer work array of length n.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Chih-Jen Lin and Jorge J. More'.
c
c     August 1999
c
c     Corrected documentation for maxgrp.
c
c     **********
      double precision zero
      parameter(zero=0.0d0)

      integer i, ip, irow, j, jp, k, l, numg, numl
      double precision sum

c     Store the i-th element of gradient difference fhesd
c     corresponding to group numgrp if there is a position
c     (i,j) such that ngrp(j) = numgrp and (i,j) is mapped
c     onto the lower triangular part of the permuted matrix.

      do j = 1, n
         if (ngrp(j) .eq. numgrp) then
            diag(j) = fhesd(j)/eta(j)
            numl = listp(j)
            do ip = row_ptr(j), row_ptr(j+1)-1
               i = col_ind(ip)
               if (listp(i) .gt. numl) then
                  do jp = col_ptr(i), col_ptr(i+1)-1
                     if (row_ind(jp) .eq. j) then
                        fhes(jp) = fhesd(i)
                        go to 10
                        end if
                  end do
   10          continue
               end if
            end do
            do jp = col_ptr(j), col_ptr(j+1)-1
               i = row_ind(jp)
               if (listp(i) .ge. numl) fhes(jp) = fhesd(i)
            end do
         end if
      end do

c     Exit if this is not the last group.

      if (numgrp .lt. maxgrp) return

c     Mark all column indices j such that (i,j) is mapped onto
c     the lower triangular part of the permuted matrix.

      do i = 1, n
         numl = listp(i)
         do ip = row_ptr(i), row_ptr(i+1)-1
            j = col_ind(ip)
            if (numl .ge. listp(j)) col_ind(ip) = -col_ind(ip)
         end do
         do jp = col_ptr(i), col_ptr(i+1)-1
            j = row_ind(jp)
            if (numl .gt. listp(j)) row_ind(jp) = -row_ind(jp)
         end do
      end do

c     Invert the array listp.

      do j = 1, n
         iwa(listp(j)) = j
      end do
      do j = 1, n
         listp(j) = iwa(j)
      end do

c     Determine the lower triangular part of the original matrix.

      do irow = n, 1, -1
         i = listp(irow)

c        Find the positions of the elements in the i-th row of the
c        lower triangular part of the original matrix that have
c        already been determined.

         do ip = row_ptr(i), row_ptr(i+1)-1
            j = col_ind(ip)
            if (j .gt. 0) then
               do jp = col_ptr(j), col_ptr(j+1)-1
                  if (row_ind(jp) .eq. i) then
                     iwa(j) = jp
                     go to 20
                  end if
               end do
   20       continue
            end if
         end do

c        Determine the elements in the i-th row of the lower
c        triangular part of the original matrix which get mapped
c        onto the lower triangular part of the permuted matrix.

         do k = row_ptr(i), row_ptr(i+1)-1
            j = -col_ind(k)
            if (j .gt. 0) then
               col_ind(k) = j

c              Determine the (i,j) element.

               numg = ngrp(j)
               sum = zero
               do ip = row_ptr(i), row_ptr(i+1)-1
                  l = abs(col_ind(ip))
                  if (ngrp(l) .eq. numg .and. l .ne. j)
     +               sum = sum + fhes(iwa(l))*eta(l)
               end do
               do jp = col_ptr(i), col_ptr(i+1)-1
                  l = abs(row_ind(jp))
                  if (ngrp(l) .eq. numg .and. l .ne. j)
     +               sum = sum + fhes(jp)*eta(l)
               end do

c              Store the (i,j) element.

               do jp = col_ptr(j), col_ptr(j+1)-1
                  if (row_ind(jp) .eq. i) then
                     fhes(jp) = (fhes(jp) - sum)/eta(j)
                     go to 30
                  end if
               end do
   30       continue
            end if
         end do


c        Determine the elements in the i-th row of the strict upper
c        triangular part of the original matrix which get mapped
c        onto the lower triangular part of the permuted matrix.

         do k = col_ptr(i), col_ptr(i+1)-1
            j = -row_ind(k)
            if (j .gt. 0) then
               row_ind(k) = j

c              Determine the (i,j) element.

               numg = ngrp(j)
               sum = zero
               do ip = row_ptr(i), row_ptr(i+1)-1
                  l = abs(col_ind(ip))
                  if (ngrp(l) .eq. numg)
     +               sum = sum + fhes(iwa(l))*eta(l)
               end do
               do jp = col_ptr(i), col_ptr(i+1)-1
                  l = abs(row_ind(jp))
                  if (ngrp(l) .eq. numg .and. l .ne. j)
     +               sum = sum + fhes(jp)*eta(l)
               end do

c              Store the (i,j) element.

               fhes(k) = (fhes(k) - sum)/eta(j)
            end if
         end do
      end do

c     Re-invert the array listp.

      do j = 1, n
         iwa(listp(j)) = j
      end do
      do j = 1, n
         listp(j) = iwa(j)
      end do

      return

      end

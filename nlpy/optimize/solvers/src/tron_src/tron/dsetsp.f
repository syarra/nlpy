      subroutine dsetsp(n,nnz,row_ind,col_ind,col_ptr,row_ptr,
     +                  method,info,listp,ngrp,maxgrp,iwa)
      integer n, nnz, method, info, maxgrp
      integer row_ind(*), col_ind(*), row_ptr(n+1), col_ptr(n+1)
      integer listp(n), ngrp(n) 
      integer iwa(*)
c     **********
c     
c     Subroutine dsetsp
c
c     Given the non-zero elements of a symmetric matrix A in coordinate
c     format, this subroutine computes the coloring information for
c     determining A from a lower triangular substitution method.
c
c     The sparsity pattern of the matrix A is specified by the
c     arrays row_ind and col_ind. On input the indices for the
c     non-zero elements in the lower triangular part of A are
c
c           (row_ind(k),col_ind(k)), k = 1,2,...,nnz.
c
c     The (row_ind(k),col_ind(k)) pairs may be specified in any order.
c     Duplicate input pairs are permitted, but they are eliminated.
c     Diagonal elements must be part of the sparsity pattern.
c     Any pair (row_ind(k),col_ind(k)), where row_ind(k) is less than 
c     col_ind(k), is replaced by the pair (col_ind(k),row_ind(k)).
c
c     The input coordinate format is changed to a storage format. 
c     On output the strict lower triangular part of A is stored in both
c     compressed column storage and compressed row storage. 
c     
c     The information required to determine the matrix from a lower 
c     triangular substitution method is obtained from subroutine dssm.
c
c     The subroutine statement is
c
c       subroutine dsetsp(n,nnz,row_ind,col_ind,col_ptr,row_ptr,
c                         method,info,listp,ngrp,maxgrp,iwa)
c
c     where
c       
c       n is an integer variable.
c         On entry n is the order of the matrix.
c         On exit n is unchanged.
c
c       nnz is an integer variable.
c         On entry nnz is the number of non-zeros entries in the
c           coordinate format.
c         On exit nnz is the number of non-zeroes in the strict
c           lower triangular part of the matrix A.
c
c       row_ind is an integer array of length nnz. 
c         On entry row_ind must contain the row indices of the non-zero 
c            elements of A in coordinate format.
c         On exit row_ind contains row indices for the strict 
c            lower triangular part of A in compressed column storage.
c
c       col_ind is an integer array of length nnz. 
c         On entry col_ind must contain the column indices of the 
c            non-zero elements of A in coordinate format. 
c         On exit col_ind contains column indices for the strict 
c            lower triangular part of A in compressed row storage.
c
c       row_ptr is an integer array of length n + 1. 
c         On entry row_ptr need not be specified.
c         On exit row_ptr must contain pointers to the rows of A.
c            The non-zeros in row j of A must be in positions
c            row_ptr(j), ... , row_ptr(j+1) - 1 of col_ind.
c
c       col_ptr is an integer array of length n + 1. 
c         On entry col_ptr need not be specified.
c         On exit col_ptr must contain pointers to the columns of A.
c            The non-zeros in column j of A must be in positions
c            col_ptr(j), ... , col_ptr(j+1) - 1 of row_ind.
c
c       method is an integer variable. 
c         On input with method = 1, the direct method is used to 
c           determine the partition and symmetric permutation. 
c           Otherwise, the indirect method is used.
c
c       info is an integer variable.
c         On input info need not be specified.
c         On output info is set as follows. 
c
c            info = 1  Normal termination.
c
c            info = 0  Input n or nnz are not positive.
c
c            info < 0  There is an error in the sparsity pattern.
c                      If k = -info then row_ind(k) or col_ind(k) is
c                      not an integer between 1 and n, or the k-th
c                      diagonal element is not in the sparsity pattern.
c
c       listp is an integer array of length n.
c         On input listp need not be specified.
c         On output listp specifies a permutation of the matrix. 
c            Element (i,j) of the matrix is the (listp(i),listp(j))
c            element of the permuted matrix.
c
c       ngrp is an integer array of length n.
c         On entry ngrp need not be specified.
c         On exit ngrp specifies the partition of the columns
c            of A. Column j belongs to group ngrp(j).
c
c       maxgrp is an integer variable. 
c         On entry maxgrp need not be specified. 
c         On exit maxgrp specifies the number of groups in
c            the partition of the columns of A.
c
c       iwa is an integer work array of length 6*n.
c
c     Subprograms called
c
c       MINPACK-2  ......  dssm
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Chih-Jen Lin and Jorge J. More'.
c
c     **********
      integer i, j, mingrp, ndiag

c     Subroutine dssm first checks the sparsity structure.
c     If there are no errors, the input format is changed to 
c     compressed column storage. Finally, the information required 
c     to determine the matrix from matrix-vector products is obtained.

      call dssm(n,nnz,row_ind,col_ind,method,listp,
     +          ngrp,maxgrp,mingrp,info,row_ptr,col_ptr,iwa,6*n)

c     Exit if there are errors on input.

      if (info .le. 0) return

c     Change the sparsity structure to exclude the diagonal entries.

      ndiag = 0
      do j = 1, n
         do i = col_ptr(j), col_ptr(j+1) - 1
            if (row_ind(i) .eq. j) then
               ndiag = ndiag + 1
            else
               row_ind(i-ndiag) =  row_ind(i)
            end if
         end do
         col_ptr(j) = col_ptr(j) - (j - 1)
      end do
      col_ptr(n+1) = col_ptr(n+1) - n
      nnz = nnz - n

      return

      end

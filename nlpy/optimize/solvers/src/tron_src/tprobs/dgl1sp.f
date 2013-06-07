      subroutine dgl1sp(n,nnz,indrow,indcol)
      integer n, nnz
      integer indrow(*), indcol(*)
c     **********
c
c     Subroutine dgl1sp
c
c     This subroutine defines the sparsity structure of the Hessian
c     matrix of the Ginzburg-Landau (1-dimensional) problem.
c
c     The subroutine statement is
c
c       subroutine dgl1sp(n,nnz,indrow,indcol)
c
c     where
c
c       n is an integer variable.
c         On entry n is the number of grid points.
c         On exit n is unchanged.
c
c       nnz is an integer variable.
c         On entry nnz need not be specified
c         On exit nnz is set to the number of nonzero index pairs
c            in the sparsity structure. Redundancy is permitted.
c
c       indrow is an integer array of dimension at least nnz.
c         On entry indrow need not be specified.
c         On exit indrow contains the row indices of the nonzeros
c            in the sparsity structure of the Hessian matrix.
c
c       indcol is an integer array of dimension at least nnz.
c         On entry indcol need not be specified.
c         On exit indcol contains the column indices of the nonzeros
c            in the sparsity structure of the Hessian matrix.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory and University of Minnesota.
c     Brett M. Averick.
c
c     **********
      integer j

c     Compute the sparsity structure.

      nnz = 0
      do 10 j = 1, n
         nnz = nnz + 1
         indrow(nnz) = j
         indcol(nnz) = j
   10 continue
      do 20 j = 1, n - 1
         nnz = nnz + 1
         indrow(nnz) = j + 1
         indcol(nnz) = j
   20 continue
      nnz = nnz + 1
      indrow(nnz) = n
      indcol(nnz) = 1

      end

      subroutine dminsp(n,nx,ny,nnz,indrow,indcol,prob)
      character*6 prob
      integer n, nx, ny, nnz
      integer indrow(*), indcol(*)
c     *********
c
c     Subroutine dminsp
c
c     This subroutine computes the sparsity pattern for
c     the minimization problem from the MINPACK-2 test problem 
c     collection specified by the character variable prob.
c
c     The subroutine statement is
c
c       dminsp(n,nx,ny,nnz,indrow,indcol,prob)
c
c     where
c
c       n is an integer variable.
c         On entry n is the number of variables.
c         On exit n is unchanged.
c
c       nx is an integer variable.
c         On entry nx is the number of grid points in the first
c            coordinate direction.
c         On exit nx is unchanged.
c
c       ny is an integer variable.
c         On entry ny is the number of grid points in the second
c            coordinate direction. If the problem is formulated in
c            one spatial dimension, ny = 1.
c         On exit ny is unchanged.
c
c       nnz is an integer variable.
c         On entry nnz need not be specified.
c         On exit nnz is set to the number of nonzeros in the
c            lower triangle of the Hessian matrix.
c
c       indrow is an integer array of dimension at least nnz.
c         On entry indrow need not be specified.
c         On exit indrow contains the row indices of the nonzeros
c           in the lower triangle of the Hessian matrix.
c
c       indcol is an integer array of dimension at least nnz.
c         On entry indcol need not be specified.
c         On exit indcol contains the column indices of the nonzeros
c            in the lower triangle of the Hessian matrix.
c
c       prob is a character*6 variable.
c         On entry prob specifies the problem.
c         On exit prob is set to 'ERROR' if prob is not an
c            acceptable problem name. Otherwise prob is unchanged.
c
c     Subprograms called
c
c       MINPACK-2 ... deptsp, dgl1sp, dgl2sp, dmsasp, dmsabc,
c                     dodcsp, dpjbsp, dsscsp
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Brett M. Averick and Jorge J. More'.
c
c     **********
      external deptsp, dgl1sp, dgl2sp, dmsasp, dodcsp, dpjbsp, dsscsp

c     Select a problem. 
     
      if (prob(1:4) .eq. 'DEPT') then
         call deptsp(nx,ny,nnz,indrow,indcol)
      else if (prob(1:4) .eq. 'DGL1') then
         call dgl1sp(n,nnz,indrow,indcol)
      else if (prob(1:4) .eq. 'DGL2') then
         call dgl2sp(nx,ny,nnz,indrow,indcol)
      else if (prob(1:4) .eq. 'DMSA') then
         call dmsasp(nx,ny,nnz,indrow,indcol)
      else if (prob(1:4) .eq. 'DMSO') then
         call dmsasp(nx,ny,nnz,indrow,indcol)
      else if (prob(1:4) .eq. 'DODC') then
         call dodcsp(nx,ny,nnz,indrow,indcol)
      else if (prob(1:4) .eq. 'DPJB') then
         call dpjbsp(nx,ny,nnz,indrow,indcol)
      else if (prob(1:4) .eq. 'DSSC') then
         call dsscsp(nx,ny,nnz,indrow,indcol)
      else
         prob = 'ERROR'
      endif
   
      end


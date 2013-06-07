      subroutine dtrpcg(ntrue,n,indfree,aprod,g,
     +                  delta,
     +                  tol,stol,itermax,w,iters,info,
     +                  wx,wy,p,q,r,t,z)
      implicit none
      integer ntrue, n, itermax, iters, info
      integer indfree(n)
      double precision delta
      double precision stol, tol
      double precision w(n), g(n)
      double precision p(n), q(n), r(n), t(n), z(n)
      double precision wx(ntrue), wy(ntrue)
      external aprod
c     *********
c     
c     Subroutine dtrpcg
c     
c     Given a sparse symmetric matrix A in compressed column storage,
c     this subroutine uses a preconditioned conjugate gradient method
c     to find an approximate minimizer of the trust region subproblem
c
c           min { q(s) : || L'*s || <= delta }.
c
c     where q is the quadratic
c
c           q(s) = 0.5*s'*A*s + g'*s,
c
c     A is a symmetric matrix in compressed column storage, L is a 
c     lower triangular matrix in compressed column storage, and g 
c     is a vector.
c
c     This subroutine generates the conjugate gradient iterates for
c     the equivalent problem
c
c           min { Q(w) : || w || <= delta }.
c
c     where Q is the quadratic defined by
c
c           Q(w) = q(s),      w = L'*s.
c
c     Termination occurs if the conjugate gradient iterates leave
c     the trust region, a negative curvature direction is generated,
c     or one of the following two convergence tests is satisfied.
c
c     Convergence in the original variables:
c
c           || grad q(s) || <= tol 
c
c     Convergence in the scaled variables:
c
c           || grad Q(w) || <= stol 
c
c     Note that if w = L'*s, then L*grad Q(w) = grad q(s).
c
c     The subroutine statement is
c
c       subroutine dtrcg(n,a,adiag,acol_ptr,arow_ind,g,delta,
c                       l,ldiag,lcol_ptr,lrow_ind,
c                       tol,stol,itermax,w,iters,info,
c                       p,q,r,t,z)
c
c     where
c
c       n is an integer variable.
c         On entry n is the number of variables.
c         On exit n is unchanged.
c
c       indfree is an integer array of dimension n.
c         On entry ia contains indices of active variables
c            in the "true" space.
c         On exit indfree is unchanged.
c
c       a is a double precision array of dimension nnz.
c         On entry a must contain the strict lower triangular part
c            of A in compressed column storage.
c         On exit a is unchanged.
c
c       adiag is a double precision array of dimension n.
c         On entry adiag must contain the diagonal elements of A.
c         On exit adiag is unchanged.
c
c       acol_ptr is an integer array of dimension n + 1.
c         On entry acol_ptr must contain pointers to the columns of A.
c            The nonzeros in column j of A must be in positions
c            acol_ptr(j), ... , acol_ptr(j+1) - 1.
c         On exit acol_ptr is unchanged.
c
c       arow_ind is an integer array of dimension nnz.
c         On entry arow_ind must contain row indices for the strict 
c            lower triangular part of A in compressed column storage.
c         On exit arow_ind is unchanged.
c
c       g is a double precision array of dimension n.
c         On entry g must contain the vector g.
c         On exit g is unchanged.
c
c       delta is a double precision variable.
c         On entry delta is the trust region size.
c         On exit delta is unchanged.
c
c       l is a double precision array of dimension nnz+n*p.
c         On entry l contains the strict lower triangular part
c            of L in compressed column storage.
c         On exit l is unchanged.
c
c       ldiag is a double precision array of dimension n.
c         On entry ldiag contains the diagonal elements of L.
c         On exit ldiag is unchanged.
c
c       lcol_ptr is an integer array of dimension n + 1.
c         On entry lcol_ptr contains pointers to the columns of L.
c            The nonzeros in column j of L are in the
c            lcol_ptr(j), ... , lcol_ptr(j+1) - 1 positions of l.
c         On exit lcol_ptr is unchanged.
c
c       lrow_ind is an integer array of dimension nnz+n*p.
c         On entry lrow_ind contains row indices for the strict lower
c            triangular part of L in compressed column storage.
c         On exit lrow_ind is unchanged.
c    
c       tol is a double precision variable.
c         On entry tol specifies the convergence test
c            in the un-scaled variables.
c         On exit tol is unchanged
c
c       stol is a double precision variable.
c         On entry stol specifies the convergence test
c            in the scaled variables.
c         On exit stol is unchanged
c
c       itermax is an integer variable.
c         On entry itermax specifies the limit on the number of
c            conjugate gradient iterations.
c         On exit itermax is unchanged.
c
c       w is a double precision array of dimension n.
c         On entry w need not be specified.
c         On exit w contains the final conjugate gradient iterate.
c
c       iters is an integer variable.
c         On entry iters need not be specified.
c         On exit iters is set to the number of conjugate 
c            gradient iterations.
c     
c       info is an integer variable.
c         On entry info need not be specified.
c         On exit info is set as follows:
c
c             info = 1  Convergence in the original variables.
c                       || grad q(s) || <= tol 
c
c             info = 2  Convergence in the scaled variables.
c                       || grad Q(w) || <= stol
c
c             info = 3  Negative curvature direction generated.
c                       In this case || w || = delta and a direction
c                       of negative curvature w can be recovered by
c                       solving L'*w = p.
c
c             info = 4  Conjugate gradient iterates exit the 
c                       trust region. In this case || w || = delta.
c
c             info = 5  Failure to converge within itermax iterations.
c
c       p is a double precision work array of dimension n.
c     
c       q is a double precision work array of dimension n.
c     
c       r is a double precision work array of dimension n.
c     
c       t is a double precision work array of dimension n.
c
c       z is a double precision work array of dimension n.
c     
c     Subprograms called
c
c       MINPACK-2  ......  dtrqsol, dstrsol
c
c       Level 1 BLAS  ...  daxpy, dcopy, ddot
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Chih-Jen Lin and Jorge J. More'.
c
c     August 1999
c
c     Corrected documentation for l, lidag, lcol_ptr and lrow_ind.
c
c     February 2001
c
c     We now set iters = 0 in the special case g = 0.
c
c     **********
      double precision zero, one
      parameter(zero=0.0d0,one=1.0d0)

      integer i
      double precision alpha, beta, ptq, rho, rtr, sigma
      double precision rnorm, rnorm0, tnorm

      double precision ddot
c     external dtrqsol, dstrsol
      external dtrqsol, asubprod
      external daxpy, dcopy, ddot

c     Initialize the iterate w and the residual r.

      do i = 1, n
         w(i) = zero
      end do

c     Initialize the residual t of grad q to -g.
c     Initialize the residual r of grad Q by solving L*r = -g.
c     Note that t = L*r.

      call dcopy(n,g,1,t,1)
      call dscal(n,-one,t,1)         
      call dcopy(n,t,1,r,1)
c     call dstrsol(n,l,ldiag,lcol_ptr,lrow_ind,r,'N')

c     Initialize the direction p.

      call dcopy(n,r,1,p,1)

c     Initialize rho and the norms of r and t.

      rho = ddot(n,r,1,r,1)
      rnorm0 = sqrt(rho)

c     Exit if g = 0.

      if (rnorm0 .eq. zero) then
         iters = 0
         info = 1
         return
      end if

      do iters = 1, itermax

c        Compute z by solving L'*z = p.

         call dcopy(n,p,1,z,1)
c        call dstrsol(n,l,ldiag,lcol_ptr,lrow_ind,z,'T')

c        Compute q by solving L*q = A*z and save L*q for
c        use in updating the residual t.

         call asubprod(aprod,indfree,ntrue,n,wx,wy,z,q)
         call dcopy(n,q,1,z,1)
c        call dstrsol(n,l,ldiag,lcol_ptr,lrow_ind,q,'N')

c        Compute alpha and determine sigma such that the trust region 
c        constraint || w + sigma*p || = delta is satisfied.

         ptq = ddot(n,p,1,q,1)
         if (ptq .gt. zero) then
            alpha = rho/ptq
         else
            alpha = zero
         end if
         call dtrqsol(n,w,p,delta,sigma)

c        Exit if there is negative curvature or if the
c        iterates exit the trust region.
         
         if (ptq .le. zero .or. alpha .ge. sigma) then
            call daxpy(n,sigma,p,1,w,1)
            if (ptq .le. zero) then
               info = 3
            else
               info = 4
            end if

            return

         end if
           
c        Update w and the residuals r and t.
c        Note that t = L*r.
         
         call daxpy(n,alpha,p,1,w,1)
         call daxpy(n,-alpha,q,1,r,1)
         call daxpy(n,-alpha,z,1,t,1)

c        Exit if the residual convergence test is satisfied.

         rtr = ddot(n,r,1,r,1) 
         rnorm = sqrt(rtr)
         tnorm = sqrt(ddot(n,t,1,t,1))

         if (tnorm .le. tol) then
            info = 1
            return
         end if

         if (rnorm .le. stol) then
            info = 2
            return
         end if

c        Compute p = r + beta*p and update rho.

         beta = rtr/rho
         call dscal(n,beta,p,1)
         call daxpy(n,one,r,1,p,1)
         rho = rtr

      end do

C     iters = itmax
      info = 5

      return

      end

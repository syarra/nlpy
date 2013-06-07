      subroutine dprsrch(ntrue,n,indfree,aprod,x,xl,xu,
     +                   g,w,wx,wy,
     +                   wa1,wa2)
      integer ntrue, n
      integer indfree(n)
      double precision x(n), xl(n), xu(n), g(n), w(n)
      double precision wa1(n), wa2(n), wx(ntrue), wy(ntrue)
      external aprod
c     **********
c
c     Subroutine dprsrch
c
c     This subroutine uses a projected search to compute a step
c     that satisfies a sufficient decrease condition for the quadratic
c
c           q(s) = 0.5*s'*A*s + g'*s,
c
c     where A is a symmetric matrix in compressed column storage,
c     and g is a vector. Given the parameter alpha, the step is
c
c           s[alpha] = P[x + alpha*w] - x,
c
c     where w is the search direction and P the projection onto the 
c     n-dimensional interval [xl,xu]. The final step s = s[alpha] 
c     satisfies the sufficient decrease condition
c
c           q(s) <= mu_0*(g'*s),
c
c     where mu_0 is a constant in (0,1).
c
c     The search direction w must be a descent direction for the
c     quadratic q at x such that the quadratic is decreasing
c     in the ray  x + alpha*w for 0 <= alpha <= 1.
c
c     The subroutine statement is
c
c       subroutine dprsrch(n,x,xl,xu,a,diag,col_ptr,row_ind,g,w, 
c                          wa1,wa2)
c
c     where
c
c       n is an integer variable.
c         On entry n is the number of variables.
c         On exit n is unchanged.
c
c       x is a double precision array of dimension n.
c         On entry x specifies the vector x.
c         On exit x is set to the final point P[x + alpha*w].
c
c       xl is a double precision array of dimension n.
c         On entry xl is the vector of lower bounds.
c         On exit xl is unchanged.
c
c       xu is a double precision array of dimension n.
c         On entry xu is the vector of upper bounds.
c         On exit xu is unchanged.
c
c       a is a double precision array of dimension nnz.
c         On entry a must contain the strict lower triangular part
c            of A in compressed column storage.
c         On exit a is unchanged.
c
c       diag is a double precision array of dimension n.
c         On entry diag must contain the diagonal elements of A.
c         On exit diag is unchanged.
c
c       col_ptr is an integer array of dimension n + 1.
c         On entry col_ptr must contain pointers to the columns of A.
c            The nonzeros in column j of A must be in positions
c            col_ptr(j), ... , col_ptr(j+1) - 1.
c         On exit col_ptr is unchanged.
c
c       row_ind is an integer array of dimension nnz.
c         On entry row_ind must contain row indices for the strict 
c            lower triangular part of A in compressed column storage.
c         On exit row_ind is unchanged.
c
c       g is a double precision array of dimension n.
c         On entry g specifies the vector g.
c         On exit g is unchanged.
c
c       w is a double prevision array of dimension n.
c         On entry w specifies the search direction.
c         On exit w is the step s[alpha].
c       
c       wa1 is a double precision work array of dimension n.
c
c       wa2 is a double precision work array of dimension n.
c
c     Subprograms called
c
c       MINPACK-2  ......  dbreakpt, dgpstep, dmid
c
c       Level 1 BLAS  ...  daxpy, dcopy, ddot
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Chih-Jen Lin and Jorge J. More'.
c
c     **********
      double precision p5, one
      parameter(p5=0.5d0,one=1.0d0)

c     Constant that defines sufficient decrease.

      double precision mu0
      parameter(mu0=0.01d0)

c     Interpolation factor.

      double precision interpf
      parameter(interpf=0.5d0)

      logical search
      integer nbrpt, nsteps
      double precision alpha, brptmin, brptmax, gts, q

      double precision ddot
      external daxpy, dcopy, ddot
      external dbreakpt, dgpstep, dmid

c     Set the initial alpha = 1 because the quadratic function is 
c     decreasing in the ray x + alpha*w for 0 <= alpha <= 1.

      alpha = one
      nsteps = 0

c     Find the smallest break-point on the ray x + alpha*w.

      call dbreakpt(n,x,xl,xu,w,nbrpt,brptmin,brptmax)

c     Reduce alpha until the sufficient decrease condition is
c     satisfied or x + alpha*w is feasible.

      search = .true.
      do while (search .and. alpha .gt. brptmin)

c        Calculate P[x + alpha*w] - x and check the sufficient
c        decrease condition.

         nsteps = nsteps + 1
         call dgpstep(n,x,xl,xu,alpha,w,wa1)
         call asubprod(aprod,indfree,ntrue,n,wx,wy,wa1,wa2)
         gts = ddot(n,g,1,wa1,1)
         q = p5*ddot(n,wa1,1,wa2,1) + gts
         if (q .le. mu0*gts) then
            search = .false.
         else

c           This is a crude interpolation procedure that
c           will be replaced in future versions of the code.

            alpha = interpf*alpha

         end if
      end do

c     Force at least one more constraint to be added to the active
c     set if alpha < brptmin and the full step is not successful. 
c     There is sufficient decrease because the quadratic function 
c     is decreasing in the ray x + alpha*w for 0 <= alpha <= 1.

      if (alpha .lt. one .and. alpha .lt. brptmin) alpha = brptmin

c     Compute the final iterate and step.

      call dgpstep(n,x,xl,xu,alpha,w,wa1)
      call daxpy(n,alpha,w,1,x,1) 
      call dmid(n,x,xl,xu)
      call dcopy(n,wa1,1,w,1)

      return

      end

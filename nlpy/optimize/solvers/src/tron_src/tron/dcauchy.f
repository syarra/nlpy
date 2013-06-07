      subroutine dcauchy(n,x,xl,xu,aprod,g,delta,
     +                   alpha,s,wa)
      integer n
      double precision delta, alpha
      double precision x(n), xl(n), xu(n), g(n), s(n)
      double precision wa(n)
      external aprod
c     **********
c
c     Subroutine dcauchy
c
c     This subroutine computes a Cauchy step that satisfies a trust
c     region constraint and a sufficient decrease condition.
c
c     The Cauchy step is computed for the quadratic
c
c           q(s) = 0.5*s'*A*s + g'*s,
c
c     where A is a symmetric matrix in compressed row storage, and
c     g is a vector. Given a parameter alpha, the Cauchy step is
c
c           s[alpha] = P[x - alpha*g] - x,
c
c     with P the projection onto the n-dimensional interval [xl,xu].
c     The Cauchy step satisfies the trust region constraint and the
c     sufficient decrease condition
c
c           || s || <= delta,      q(s) <= mu_0*(g'*s),
c
c     where mu_0 is a constant in (0,1).
c
c     The subroutine statement is
c
c       subroutine dcauchy(n,x,xl,xu,a,diag,col_ptr,row_ind,g,delta, 
c                          alpha,s,wa)
c
c     where
c
c
c       n is an integer variable.
c         On entry n is the number of variables.
c         On exit n is unchanged.
c
c       x is a double precision array of dimension n.
c         On entry x specifies the vector x.
c         On exit x is unchanged.
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
c         On entry g specifies the gradient g.
c         On exit g is unchanged.
c
c       delta is a double precision variable.
c         On entry delta is the trust region size.
c         On exit delta is unchanged.
c
c       alpha is a double precision variable.
c         On entry alpha is the current estimate of the step.
c         On exit alpha defines the Cauchy step s[alpha].
c
c       s is a double precision array of dimension n.
c         On entry s need not be specified.
c         On exit s is the Cauchy step s[alpha].
c
c       wa is a double precision work array of dimension n.
c
c     Subprograms called
c
c       MINPACK-2  ......  dbreakpt, dgpstep
c
c       Level 1 BLAS  ...  ddot, dnrm2
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Chih-Jen Lin and Jorge J. More'.
c
c     February 2000
c
c     Eliminated the nsteps variable.
c     Made sure that variable alphas was initialized.
c     Corrected the calculation of the minimal and maximal breakpoints
c     by replacing g with wa in the call to breakpt.
c
c     **********
      double precision p5, one
      parameter(p5=0.5d0,one=1.0d0)

c     Constant that defines sufficient decrease.

      double precision mu0
      parameter(mu0=0.01d0)

c     Interpolation and extrapolation factors.

      double precision interpf, extrapf
      parameter(interpf=0.1d0,extrapf=1.0d1)

      logical search, interp
      integer nbrpt
      double precision alphas, brptmax, brptmin, gts, q

      double precision dnrm2, ddot
      external dnrm2, ddot

c     Find the minimal and maximal break-point on x - alpha*g.

      call dcopy(n,g,1,wa,1)
      call dscal(n,-one,wa,1)
      call dbreakpt(n,x,xl,xu,wa,nbrpt,brptmin,brptmax)

c     Evaluate the initial alpha and decide if the algorithm
c     must interpolate or extrapolate.

      call dgpstep(n,x,xl,xu,-alpha,g,s)
      if (dnrm2(n,s,1) .gt. delta) then
         interp = .true.
      else 
         call aprod(n,s,wa)
         gts = ddot(n,g,1,s,1)
         q = p5*ddot(n,s,1,wa,1) + gts
         interp = (q .ge. mu0*gts) 
      end if

c     Either interpolate or extrapolate to find a successful step.

      if (interp) then

c        Reduce alpha until a successful step is found.

         search = .true.
         do while (search)

c           This is a crude interpolation procedure that
c           will be replaced in future versions of the code.

            alpha = interpf*alpha
            call dgpstep(n,x,xl,xu,-alpha,g,s)
            if (dnrm2(n,s,1) .le. delta) then
               call aprod(n,s,wa)
               gts = ddot(n,g,1,s,1)
               q = p5*ddot(n,s,1,wa,1) + gts
               search = (q .gt. mu0*gts)
            end if
         end do

      else

c        Increase alpha until a successful step is found.

         search = .true.
         alphas = alpha
         do while (search .and. alpha .le. brptmax)

c           This is a crude extrapolation procedure that
c           will be replaced in future versions of the code.

            alpha = extrapf*alpha
            call dgpstep(n,x,xl,xu,-alpha,g,s)
            if (dnrm2(n,s,1) .le. delta) then
               call aprod(n,s,wa)
               gts = ddot(n,g,1,s,1)
               q = p5*ddot(n,s,1,wa,1) + gts
               if (q .lt. mu0*gts) then
                  search = .true.
                  alphas = alpha
               end if
            else
               search = .false.
            end if
         end do

c        Recover the last successful step.

         alpha = alphas
         call dgpstep(n,x,xl,xu,-alpha,g,s)
      end if

      return

      end

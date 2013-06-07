      subroutine dtron(n,x,xl,xu,f,g,aprod,
     +                 frtol,fatol,fmin,cgtol,itermax,delta,task,
     +                 xc,s,indfree,
     +                 isave,dsave,wa,wx,wy,iwa)
      character*60 task
      integer n, itermax
      integer indfree(n)
      integer isave(3)
      integer iwa(3*n)
      double precision f, frtol, fatol, fmin, cgtol, delta
      double precision x(n), xl(n), xu(n), g(n), xc(n), s(n)
      double precision dsave(3) 
      double precision wa(7*n)
      double precision wx(n), wy(n)
c     *********
c
c     Subroutine dtron
c
c     This subroutine implements a trust region Newton method for the
c     solution of large bound-constrained optimization problems
c
c           min { f(x) : xl <= x <= xu }
c
c     where the Hessian matrix is sparse. The user must evaluate the
c     function, gradient, and the Hessian matrix.
c
c     This subroutine uses reverse communication.
c     The user must choose an initial approximation x to the minimizer,
c     and make an initial call with task set to 'START'.
c     On exit task indicates the required action.
c
c     A typical invocation has the following outline:
c
c     Compute a starting vector x.
c     Compute the sparsity pattern of the Hessian matrix and
c     store in compressed column storage in (acol_ptr,arow_ind).
C
c     task = 'START'
c     do while (search) 
c
c        if (task .eq. 'F' .or. task .eq. 'START') then
c           Evaluate the function at x and store in f.
c        end if
c        if (task .eq. 'GH' .or. task .eq. 'START') then
c           Evaluate the gradient at x and store in g.
c           Evaluate the Hessian at x and store in compressed
c           column storage in (a,adiag,acol_ptr,arow_ind)
c        end if
c
c        call dtron(n,x,xl,xu,f,g,a,adiag,acol_ptr,arow_ind,
c                   frtol,fatol,fmin,cgtol,itermax,delta,task,
c                   b,bdiag,bcol_ptr,brow_ind,
c                   l,ldiag,lcol_ptr,lrow_ind,
c                   xc,s,indfree,
c                   isave,dsave,wa,wb,iwa)
c
c        if (task(1:4) .eq. 'CONV') search = .false.
c
c      end do
c
c     NOTE: The user must not alter work arrays between calls.
c
c     The subroutine statement is
c
c       subroutine dtron(n,x,xl,xu,f,g,a,adiag,acol_ptr,arow_ind,
c                        frtol,fatol,fmin,cgtol,itermax,delta,task,
c                        b,bdiag,bcol_ptr,brow_ind,
c                        l,ldiag,lcol_ptr,lrow_ind,
c                        xc,s,indfree,
c                        isave,dsave,wa,iwa)
c
c     where
c
c       n is an integer variable.
c         On entry n is the number of variables.
c         On exit n is unchanged.
c
c       x is a double precision array of dimension n.
c         On entry x specifies the vector x.
c         On exit x is the final minimizer.
c
c       xl is a double precision array of dimension n.
c         On entry xl is the vector of lower bounds.
c         On exit xl is unchanged.
c
c       xu is a double precision array of dimension n.
c         On entry xu is the vector of upper bounds.
c         On exit xu is unchanged.
c
c       f is a double precision variable.
c         On entry f must contain the function at x.
c         On exit f is unchanged.
c
c       g is a double precision array of dimension n.
c         On entry g must contain the gradient at x.
c         On exit g is unchanged.
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
c       frtol is a double precision variable.
c         On entry frtol specifies the relative error desired in the
c            function. Convergence occurs if the estimate of the
c            relative error between f(x) and f(xsol), where xsol
c            is a local minimizer, is less than frtol.
c         On exit frtol is unchanged.
c
c       fatol is a double precision variable.
c         On entry fatol specifies the absolute error desired in the
c            function. Convergence occurs if the estimate of the
c            absolute error between f(x) and f(xsol), where xsol
c            is a local minimizer, is less than fatol.
c         On exit fatol is unchanged.
c
c       fmin is a double precision variable.
c         On entry fmin specifies a lower bound for the function.
c            The subroutine exits with a warning if f < fmin.
c         On exit fmin is unchanged.
c
c       cgtol is a double precision variable.
c         On entry cgtol specifies the convergence criteria for
c            the conjugate gradient method.
c         On exit cgtol is unchanged.
c
c       itermax is an integer variable.
c         On entry itermax specifies the limit on the number of
c            conjugate gradient iterations.
c         On exit itermax is unchanged.
c
c       delta is a double precision variable.
c         On entry delta is the trust region bound.
c         On exit delta is unchanged.
c
c       task is a character variable of length at least 60.
c         On initial entry task must be set to 'START'.
c         On exit task indicates the required action:
c
c            If task(1:1) = 'F' then evaluate the function at x.
c
c            If task(1:2) = 'GH' then evaluate the gradient and the
c            Hessian matrix at x.
c
c            If task(1:4) = 'CONV' then the search is successful.
c
c            If task(1:4) = 'WARN' then the subroutine is not able
c            to satisfy the convergence conditions. The exit value
c            of x contains the best approximation found.
c
c       bdiag is a double precision array of dimension n.
c         On entry bdiag need not be specified.
c         On exit bdiag contains the diagonal elements of B.
c
c       bcol_ptr is an integer array of dimension n + 1.
c         On entry bcol_ptr need not be specified
c         On exit bcol_ptr contains pointers to the columns of B.
c            The nonzeros in column j of B are in the
c            bcol_ptr(j), ... , bcol_ptr(j+1) - 1 positions of b.
c
c       brow_ind is an integer array of dimension nnz.
c         On entry brow_ind need not be specified.
c         On exit brow_ind contains row indices for the strict lower
c            triangular part of B in compressed column storage. 
c
c       l is a double precision array of dimension nnz + n*p.
c         On entry l need not be specified.
c         On exit l contains the strict lower triangular part
c            of L in compressed column storage.
c
c       ldiag is a double precision array of dimension n.
c         On entry ldiag need not be specified.
c         On exit ldiag contains the diagonal elements of L.
c
c       lcol_ptr is an integer array of dimension n + 1.
c         On entry lcol_ptr need not be specified.
c         On exit lcol_ptr contains pointers to the columns of L.
c            The nonzeros in column j of L are in the
c            lcol_ptr(j), ... , lcol_ptr(j+1) - 1 positions of l.
c
c       lrow_ind is an integer array of dimension nnz + n*p.
c         On entry lrow_ind need not be specified.
c         On exit lrow_ind contains row indices for the strict lower
c            triangular part of L in compressed column storage. 
c
c       xc is a double precision working array of dimension n.
c
c       s is a double precision working array of dimension n.
c
c       indfree is an integer working array of dimension n.
c
c       isave is an integer working array of dimension 3.
c
c       dsave is  double precision working array of dimension 3.
c
c       wa is a double precision work array of dimension 7*n.
c
c       iwa is an integer work array of dimension 3*n.
c
c     Subprograms called
c
c       MINPACK-2  ......  dcauchy, dspcg
c
c       Level 1 BLAS  ...  dcopy
c
c     MINPACK-2 Project. May 1999.
c     Argonne National Laboratory.
c     Chih-Jen Lin and Jorge J. More'.
c
c     **********
      double precision zero, p5, one
      parameter(zero=0.0d0,p5=0.5d0,one=1.0d0)

c     Parameters for updating the iterates.

      double precision eta0, eta1, eta2
      parameter(eta0=1d-4,eta1=0.25d0,eta2=0.75d0)

c     Parameters for updating the trust region size delta.

      double precision sigma1, sigma2, sigma3
      parameter(sigma1=0.25d0,sigma2=0.5d0,sigma3=4.0d0)

      logical search
      character*60 work
      integer iter, iters, iterscg
      double precision alphac, fc, prered, actred, snorm, g0, alpha

      double precision ddot, dnrm2
      external dcauchy, dspcg
      external aprod
      external dcopy, ddot, dnrm2

c     Initialization section.

      if (task(1:5) .eq. 'START') then

c        Initialize local variables.

         iter = 1
         iterscg = 0
         alphac = one
         work = 'COMPUTE'

      else

c        Restore local variables.

         if (isave(1) .eq. 1) then
            work = 'COMPUTE' 
         else if (isave(1) .eq. 2) then
            work = 'EVALUATE'
         else if (isave(1) .eq. 3) then
            work = 'NEWX'
         end if
         iter = isave(2)
         iterscg = isave(3)
         fc = dsave(1) 
         alphac = dsave(2)
         prered = dsave(3) 
      end if

c     Search for a lower function value.

      search = .true.
      do while (search)

c        Compute a step and evaluate the function at the trial point.

         if (work .eq. 'COMPUTE') then
         
c           Save the best function value, iterate, and gradient.
         
            fc = f
            call dcopy(n,x,1,xc,1)

c           Compute the Cauchy step and store in s.
         
            call dcauchy(n,x,xl,xu,aprod,g,delta,
     +                   alphac,s,wa)
         
c           Compute the projected Newton step.

            call dspcg(n,x,xl,xu,aprod,g,
     +                 delta,cgtol,s,itermax,iters,info,
     +                 indfree,wa(1),
     +                 wa(n+1),wa(2*n+1),wx,wy,iwa)

c           Compute the predicted reduction.
            
            call aprod(n,s,wa)
            prered = -(ddot(n,s,1,g,1) + p5*ddot(n,s,1,wa,1))
            iterscg = iterscg + iters

c           Set task to compute the function.

            task = 'F'

         end if

c        Evaluate the step and determine if the step is successful.
         
         if (work .eq. 'EVALUATE') then

c           Compute the actual reduction. 
         
            actred =  fc - f

c           On the first iteration, adjust the initial step bound.

            snorm = dnrm2(n,s,1)

            if (iter .eq. 1)  delta = min(delta,snorm)
         
c           Update the trust region bound.
         
            g0 = ddot(n,g,1,s,1)
            if (f-fc-g0 .le. zero) then 
               alpha = sigma3 
            else 
               alpha = max(sigma1,-p5*(g0/(f-fc-g0))) 
            end if

c           Update the trust region bound according to the ratio
c           of actual to predicted reduction.

            if (actred .lt. eta0*prered) then
               delta = min(max(alpha,sigma1)*snorm,sigma2*delta)
            else if (actred .lt. eta1*prered) then
               delta = max(sigma1*delta,min(alpha*snorm,sigma2*delta))
            else if (actred .lt. eta2*prered) then
               delta = max(sigma1*delta,min(alpha*snorm,sigma3*delta))
            else
               delta = max(delta,min(alpha*snorm,sigma3*delta))
            end if

c           Update the iterate. 
         
            if (actred .gt. eta0*prered) then

c              Successful iterate.
               task = 'GH'
               iter = iter + 1

            else

c              Unsuccessful iterate.
               task = 'F'
               call dcopy(n,xc,1,x,1)
               f = fc

            end if
         
c           Test for convergence.
         
            if (f .lt. fmin) task = 'WARNING: F .LT. FMIN'
            if (abs(actred) .le. fatol .and. prered .le. fatol) task =
     +          'CONVERGENCE: FATOL TEST SATISFIED'
            if (abs(actred) .le. frtol*abs(f) .and.
     +          prered .le. frtol*abs(f)) task =
     +          'CONVERGENCE: FRTOL TEST SATISFIED'
         
         end if

c        Test for continuation of search

        if (task .eq. 'F' .and. work .eq. 'EVALUATE') then
           search = .true.
           work = 'COMPUTE'
        else
           search = .false.
        end if
         
      end do

      if (work .eq. 'NEWX') task = 'NEWX'

c     Decide on what work to perform on the next iteration.

      if (task .eq. 'F' .and. work .eq. 'COMPUTE') then
         work = 'EVALUATE'
      else if (task .eq. 'F' .and. work .eq. 'EVALUATE') then
         work = 'COMPUTE'
      else if (task .eq. 'GH') then
         work = 'NEWX'
      else if (task .eq. 'NEWX') then
         work = 'COMPUTE'
      end if

c     Save local variables.

      if (work .eq. 'COMPUTE') then
         isave(1) = 1
      else if (work .eq. 'EVALUATE') then
         isave(1) = 2
      else if (work .eq. 'NEWX') then
        isave(1) = 3
      end if
      isave(2) = iter 
      isave(3) = iterscg

      dsave(1) = fc
      dsave(2) = alphac
      dsave(3) = prered

      end

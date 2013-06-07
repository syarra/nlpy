      subroutine dodcps(t,mu1,mu2,t1,t2,result,option,lambda)
      integer option
      double precision t, mu1, mu2, t1, t2, result, lambda
c     **********
c
c     This subroutine computes the function psi(t) and the scaled
c     functions psi'(t)/t and psi''(t)/t for the optimal design
c     with composite materials problem.
c
c     The subroutine statement is
c
c       subroutine dodcps(t,mu1,mu2,t1,t2,result,option,lambda)
c
c     where
c
c       t is a double precision variable.
c         On entry t is the variable t
c         On exit t is unchanged
c
c       mu1 is a double precision variable.
c         On entry mu1 is the reciprocal shear modulus of material 1.
c         On exit mu1 is unchanged.
c
c       mu2 is a double precision variable.
c         On entry mu2 is the reciprocal shear modulus of material 2.
c         On exit mu2 is unchanged.
c
c       t1 is a double precision variable.
c         On entry t1 is the first breakpoint.
c         On exit t1 is unchanged.
c
c       t2 is a double precision variable.
c         On entry t2 is the second breakpoint.
c         On exit t2 is unchanged.
c
c       result is a double precision variable.
c         On entry result need not be specified.
c         On exit result is set according to task.
c
c       option is an integer variable.
c         On entry option specifies the action of the subroutine:
c
c            if option = 0 then evaluate the function psi(t).
c            if option = 1 then evaluate the scaled function psi'(t)/t.
c            if option = 2 then evaluate the scaled function psi''(t)/t.
c
c        On option task is unchanged.
c
c       lambda is a double precision variable
c         On entry lambda is the Lagrange multiplier.
c         On exit lambda is unchanged.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory and University of Minnesota.
c     Brett M. Averick.
c
c     **********
      double precision p25, p5, zero
      parameter (zero=0.0d0,p25=0.25d0,p5=0.5d0)

      double precision sqrtt

      sqrtt = sqrt(t)

      if (option .eq. 0) then
         if (sqrtt .le. t1) then
            result = p5*mu2*t
         else if (sqrtt .gt. t1 .and. sqrtt .lt. t2) then
            result = mu2*t1*sqrtt - lambda*mu1
         else if (sqrtt .ge. t2) then
            result = p5*mu1*t + lambda*(mu2-mu1)
         end if
      else if (option .eq. 1) then
         if (sqrtt .le. t1) then
            result = p5*mu2
         else if (sqrtt .gt. t1 .and. sqrtt .lt. t2) then
            result = p5*mu2*t1/sqrtt
         else if (sqrtt .ge. t2) then
            result = p5*mu1
         end if
      else if (option .eq. 2) then
         if (sqrtt .le. t1) then
            result = zero
         else if (sqrtt .gt. t1 .and. sqrtt .lt. t2) then
            result = -p25*mu2*t1/(sqrtt*t)
         else if (sqrtt .ge. t2) then
            result = zero
         end if
      end if

      end

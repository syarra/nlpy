      subroutine dtrqsol(n,x,p,delta,sigma)
      integer n
      double precision delta, sigma
      double precision x(n), p(n)
c     **********
c
c     Subroutine dtrqsol
c
c     This subroutine computes the largest (non-negative) solution
c     of the quadratic trust region equation
c
c           ||x + sigma*p|| = delta.
c
c     The code is only guaranteed to produce a non-negative solution
c     if ||x|| <= delta, and p != 0. If the trust region equation has 
c     no solution, sigma = 0.
c
c     The subroutine statement ix
c
c       dtrqsol(n,x,p,delta,sigma)
c
c     where
c
c       n is an integer variable.
c         On entry n is the number of variables.
c         On exit n is unchanged.
c
c       x is a double precision array of dimension n.
c         On entry x must contain the vector x.
c         On exit x is unchanged.
c
c       p is a double precision array of dimension n.
c         On entry p must contain the vector p.
c         On exit p is unchanged.
c
c       delta is a double precision variable.
c         On entry delta specifies the scalar delta.
c         On exit delta is unchanged.
c
c       sigma is a double precision variable.
c         On entry sigma need not be specified.
c         On exit sigma contains the non-negative solution.
c
c     Subprograms called
c
c       Level 1 BLAS  ...  ddot 
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Chih-Jen Lin and Jorge J. More'.
c
c     **********
      double precision zero
      parameter(zero=0.0d0)

      double precision dsq, ptp, ptx, rad, xtx

      double precision ddot

      ptx = ddot(n,p,1,x,1)
      ptp = ddot(n,p,1,p,1)
      xtx = ddot(n,x,1,x,1)
      dsq = delta**2

c     Guard against abnormal cases.

      rad = ptx**2 + ptp*(dsq - xtx)
      rad = sqrt(max(rad,zero))

      if (ptx .gt. zero) then
         sigma = (dsq - xtx)/(ptx + rad)
      else if (rad .gt. zero) then
         sigma = (rad - ptx)/ptp
      else
         sigma = zero
      end if

      return

      end

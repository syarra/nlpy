      double precision function dgpnrm2(n,x,xl,xu,g)
      integer n
      double precision x(n), xl(n), xu(n), g(n)
c     **********
c
c     Function dgpnrm2
c
c     This function computes the Euclidean norm of the
c     projected gradient at x.
c
c     The function statement is
c
c       double precision function dgpnrm2(n,x,xl,xu,g)
c
c     where
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
c       g is a double precision array of dimension n.
c         On entry g specifies the gradient g.
c         On exit g is unchanged.
c
c     MINPACK-2 Project. May 1999.
c     Argonne National Laboratory.
c     Chih-Jen Lin and Jorge J. More'.
c
c     **********
      double precision zero
      parameter (zero=0.0d0)

      integer i

      dgpnrm2 = zero
      do i = 1, n
         if (xl(i) .ne. xu(i)) then
            if (x(i) .eq. xl(i)) then
               dgpnrm2 = dgpnrm2 + min(g(i),zero)**2
            else if (x(i) .eq. xu(i)) then
               dgpnrm2 = dgpnrm2 + max(g(i),zero)**2
            else
               dgpnrm2 = dgpnrm2 + g(i)**2
            end if
         end if
      end do
      dgpnrm2 = sqrt(dgpnrm2)

      return

      end

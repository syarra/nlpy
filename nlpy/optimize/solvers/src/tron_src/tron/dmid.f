      subroutine dmid(n,x,xl,xu) 
      integer n 
      double precision x(n), xl(n), xu(n)
c     **********
c
c     Subroutine dmid
c
c     This subroutine computes the projection of x
c     on the n-dimensional interval [xl,xu].
c
c     The subroutine statement is
c
c       subroutine dmid(n,x,xl,xu)
c
c     where
c
c       n is an integer variable.
c         On entry n is the number of variables.
c         On exit n is unchanged.
c
c       x is a double precision array of dimension n.
c         On entry x specifies the vector x.
c         On exit x is the projection of x on [xl,xu].
c
c       xl is a double precision array of dimension n.
c         On entry xl is the vector of lower bounds.
c         On exit xl is unchanged.
c
c       xu is a double precision array of dimension n.
c         On entry xu is the vector of upper bounds.
c         On exit xu is unchanged.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Chih-Jen Lin and Jorge J. More'.
c
c     **********
      integer i

      do i = 1, n
         x(i) = max(xl(i),min(x(i),xu(i)))
      end do

      return

      end


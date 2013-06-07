      subroutine dbreakpt(n,x,xl,xu,w,nbrpt,brptmin,brptmax)
      integer n, nbrpt
      double precision brptmin, brptmax
      double precision x(n), xl(n), xu(n), w(n)
c     **********
c
c     Subroutine dbreakpt
c
c     This subroutine computes the number of break-points, and
c     the minimal and maximal break-points of the projection of 
c     x + alpha*w on the n-dimensional interval [xl,xu].
c
c     The subroutine statement is
c
c       subroutine dbreakpt(n,x,xl,xu,w,nbrpt,brptmin,brptmax)
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
c       w is a double precision array of dimension n.
c         On entry w specifies the vector w.
c         On exit w is unchanged.
c
c       nbrpt is an integer variable.
c         On entry nbrpt need not be specified.
c         On exit nbrpt is the number of break points.
c
c       brptmin is a double precision variable
c         On entry brptmin need not be specified.
c         On exit brptmin is minimal break-point.
c
c       brptmax is a double precision variable
c         On entry brptmax need not be specified.
c         On exit brptmax is maximal break-point.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Chih-Jen Lin and Jorge J. More'.
c
c     **********
      double precision zero
      parameter(zero=0.0d0)

      integer i
      double precision brpt

      nbrpt = 0 
      do i = 1, n
         if (x(i) .lt. xu(i) .and. w(i) .gt. zero) then 
            nbrpt = nbrpt + 1
            brpt =  (xu(i) - x(i))/w(i)
            if (nbrpt .eq. 1) then
               brptmin = brpt
               brptmax = brpt
            else
               brptmin = min(brpt,brptmin)
               brptmax = max(brpt,brptmax)
            end if
         else if (x(i) .gt. xl(i) .and. w(i) .lt. zero) then
            nbrpt = nbrpt + 1
            brpt = (xl(i) - x(i))/w(i)
            if (nbrpt .eq. 1) then
               brptmin = brpt
               brptmax = brpt
            else
               brptmin = min(brpt,brptmin)
               brptmax = max(brpt,brptmax)
            end if
         end if
      end do

c     Handle the exceptional case.

      if (nbrpt .eq. 0) then
         brptmin = zero
         brptmax = zero
      end if

      return

      end

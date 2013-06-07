      subroutine dmsabc(nx,ny,bottom,top,left,right)
      integer nx, ny
      double precision bottom(nx+2), top(nx+2), left(ny+2), right(ny+2)
c     **********
c
c     Subroutine dmsabc
c
c     This subroutine computes Enneper's boundary conditions for the
c     minimal surface area problem on the unit square centered at the
c     origin.
c
c     The subroutine statement is
c
c       subroutine dmsabc(nx,ny,bottom,top,left,right)
c
c     where
c
c       nx is an integer variable.
c         On entry nx is the number of grid points in the first
c            coordinate direction.
c         On exit nx is unchanged.
c
c       ny is an integer variable.
c         On entry ny is the number of grid points in the second
c            coordinate direction.
c         On exit ny is unchanged.
c
c       bottom is a double precision array of dimension nx + 2.
c         On entry bottom need not be specified.
c         On exit bottom contains boundary values for the bottom
c            boundary of the domain.
c
c       top is a double precision array of dimension nx + 2.
c         On entry top need not be specified.
c         On exit top contains boundary values for the top boundary of
c            the domain.
c       left is a double precision array of dimension ny + 2.
c         On entry left need not be specified.
c         On exit left contains boundary values for the left boundary
c            of the domain.
c
c       right is a double precision array of dimension ny + 2.
c         On entry right need not be specified.
c         On exit right contains boundary values for the right boundary
c            of the domain.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory and University of Minnesota.
c     Brett M. Averick.
c
c     **********
      integer maxit
      double precision b, l, one, r, t, three, tol, two
      parameter (one=1.0d0,two=2.0d0,three=3.0d0)
      parameter (maxit=5,tol=1.0d-10)
      parameter (b=-.50d0,t=.50d0,l=-.50d0,r=.50d0)

      integer i, j, k, limit
      double precision det, fnorm, hx, hy, xt, yt
      double precision nf(2), njac(2,2), u(2)

c     Compute Enneper's boundary conditions: bottom, top, left, then
c     right.  Enneper's boundary values are obtained by defining
c     bv(x,y) = u**2 - v**2 where u and v are the unique solutions of
c     x = u + u*(v**2) - (u**3)/3, y = -v - (u**2)*v + (v**3)/3.

      hx = (r-l)/dble(nx+1)
      hy = (t-b)/dble(ny+1)

      do 40 j = 1, 4
         if (j .eq. 1) then
            yt = b
            xt = l
            limit = nx + 2
         else if (j .eq. 2) then
            yt = t
            xt = l
            limit = nx + 2
         else if (j .eq. 3) then
            yt = b
            xt = l
            limit = ny + 2
         else if (j .eq. 4) then
            yt = b
            xt = r
            limit = ny + 2
         end if

c        Use Newton's method to solve xt = u + u*(v**2) - (u**3)/3,
c        yt = -v - (u**2)*v + (v**3)/3.

         do 30 i = 1, limit
            u(1) = xt
            u(2) = -yt
            do 10 k = 1, maxit
               nf(1) = u(1) + u(1)*u(2)**2 - u(1)**3/three - xt
               nf(2) = -u(2) - u(1)**2*u(2) + u(2)**3/three - yt
               fnorm = sqrt(nf(1)*nf(1)+nf(2)*nf(2))
               if (fnorm .le. tol) go to 20
               njac(1,1) = one + u(2)**2 - u(1)**2
               njac(1,2) = two*u(1)*u(2)
               njac(2,1) = -two*u(1)*u(2)
               njac(2,2) = -one - u(1)**2 + u(2)**2
               det = njac(1,1)*njac(2,2) - njac(1,2)*njac(2,1)
               u(1) = u(1) - (njac(2,2)*nf(1)-njac(1,2)*nf(2))/det
               u(2) = u(2) - (njac(1,1)*nf(2)-njac(2,1)*nf(1))/det
   10       continue
   20       continue

            if (j .eq. 1) then
               bottom(i) = u(1)*u(1) - u(2)*u(2)
               xt = xt + hx
            else if (j .eq. 2) then
               top(i) = u(1)*u(1) - u(2)*u(2)
               xt = xt + hx
            else if (j .eq. 3) then
               left(i) = u(1)*u(1) - u(2)*u(2)
               yt = yt + hy
            else if (j .eq. 4) then
               right(i) = u(1)*u(1) - u(2)*u(2)
               yt = yt + hy
            end if
   30    continue
   40 continue

      end

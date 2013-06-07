      subroutine dpjbfg(nx,ny,x,f,fgrad,task,ecc,b)
      character*60 task
      integer nx, ny
      double precision f, ecc, b
      double precision x(nx*ny), fgrad(nx*ny)
c     **********
c
c     Subroutine dpjbfg
c
c     This subroutine computes the function and gradient of the
c     pressure distribution in a journal bearing problem.
c
c     The subroutine statement is
c
c       subroutine dpjbfg(nx,ny,x,f,fgrad,task,ecc,b)
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
c       x is a double precision array of dimension nx*ny.
c         On entry x specifies the vector x if task = 'F', 'G', or 'FG'.
c            Otherwise x need not be specified.
c         On exit x is unchanged if task = 'F', 'G', or 'FG'. Otherwise
c            x is set according to task.
c
c       f is a double precision variable.
c         On entry f need not be specified.
c         On exit f is set to the function evaluated at x if task = 'F'
c            or 'FG'.
c
c       fgrad is a double precision array of dimension nx*ny.
c         On entry fgrad need not be specified.
c         On exit fgrad contains the gradient evaluated at x if
c            task = 'G' or 'FG'.
c
c       task is a character*60 variable.
c         On entry task specifies the action of the subroutine:
c
c            task               action
c            ----               ------
c             'F'     Evaluate the function at x.
c             'G'     Evaluate the gradient at x.
c             'FG'    Evaluate the function and the gradient at x.
c             'XS'    Set x to the standard starting point xs.
c             'XL'    Set x to the lower bound xl.
c
c         On exit task is unchanged.
c
c       ecc is a double precision variable
c         On entry ecc is the eccentricity in (0,1).
c         On exit ecc is unchanged
c
c       b is a double precision variable
c         On entry b defines the domain as D = (0,2*pi) X (0,2*b).
c         On exit b is unchanged.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory and University of Minnesota.
c     Brett M. Averick and Jorge J. More'.
c
c     **********
      double precision four, one, p5, six, two, zero
      parameter (zero=0.0d0,p5=0.5d0,one=1.0d0,two=2.0d0,four=4.0d0,
     +          six=6.0d0)

      logical feval, geval
      integer i, j, k
      double precision dvdx, dvdy, ehxhy, flin, fquad, hx, hxhy, hy, pi,
     +                 temp, trule, v, vb, vl, vr, vt, xi

      double precision p

      p(xi) = (1+ecc*cos(xi))**3

c     Initialization.

      pi = four*atan(one)
      hx = two*pi/dble(nx+1)
      hy = two*b/dble(ny+1)
      hxhy = hx*hy
      ehxhy = ecc*hxhy

c     Compute the lower bound xl for x if task = 'XL'.

      if (task(1:2) .eq. 'XL') then
         do 10 k = 1, nx*ny
            x(k) = zero
   10    continue

         return

      end if

c     Compute the standard starting point if task = 'XS'.

      if (task(1:2) .eq. 'XS') then
         do 30 i = 1, nx
            temp = max(sin(dble(i)*hx),zero)
            do 20 j = 1, ny
               k = nx*(j-1) + i
               x(k) = temp
   20       continue
   30    continue

         return

      end if

      if (task(1:1) .eq. 'F' .or. task(1:2) .eq. 'FG') then
         feval = .true.
      else
         feval = .false.
      end if
      if (task(1:1) .eq. 'G' .or. task(1:2) .eq. 'FG') then
         geval = .true.
      else
         geval = .false.
      end if

c     Compute the function if task = 'F', the gradient if task = 'G', or
c     both if task = 'FG'.

      if (feval) then
         fquad = zero
         flin = zero
      end if
      if (geval) then
         do 40 k = 1, nx*ny
            fgrad(k) = zero
   40    continue
      end if

c     Computation of the quadratic part of the function and
c     corresponding components of the gradient over the
c     lower triangular elements.

      do 60 i = 0, nx
         xi = dble(i)*hx
         trule = hxhy*(p(xi)+p(xi+hx)+p(xi))/six
         do 50 j = 0, ny
            k = nx*(j-1) + i
            v = zero
            vr = zero
            vt = zero
            if (i .ne. 0 .and. j .ne. 0) v = x(k)
            if (i .ne. nx .and. j .ne. 0) vr = x(k+1)
            if (i .ne. 0 .and. j .ne. ny) vt = x(k+nx)
            dvdx = (vr-v)/hx
            dvdy = (vt-v)/hy
            if (feval) fquad = fquad + trule*(dvdx**2+dvdy**2)
            if (geval) then
               if (i .ne. 0 .and. j .ne. 0)
     +             fgrad(k) = fgrad(k) - trule*(dvdx/hx+dvdy/hy)
               if (i .ne. nx .and. j .ne. 0)
     +             fgrad(k+1) = fgrad(k+1) + trule*dvdx/hx
               if (i .ne. 0 .and. j .ne. ny)
     +             fgrad(k+nx) = fgrad(k+nx) + trule*dvdy/hy
            end if
   50    continue
   60 continue

c     Computation of the quadratic part of the function and
c     corresponding components of the gradient over the upper
c     triangular elements.

      do 80 i = 1, nx + 1
         xi = dble(i)*hx
         trule = hxhy*(p(xi)+p(xi-hx)+p(xi))/six
         do 70 j = 1, ny + 1
            k = nx*(j-1) + i
            vb = zero
            vl = zero
            v = zero
            if (i .ne. nx+1 .and. j .ne. 1) vb = x(k-nx)
            if (i .ne. 1 .and. j .ne. ny+1) vl = x(k-1)
            if (i .ne. nx+1 .and. j .ne. ny+1) v = x(k)
            dvdx = (v-vl)/hx
            dvdy = (v-vb)/hy
            if (feval) fquad = fquad + trule*(dvdx**2+dvdy**2)
            if (geval) then
               if (i .le. nx .and. j .gt. 1)
     +             fgrad(k-nx) = fgrad(k-nx) - trule*dvdy/hy
               if (i .gt. 1 .and. j .le. ny)
     +             fgrad(k-1) = fgrad(k-1) - trule*dvdx/hx
               if (i .le. nx .and. j .le. ny)
     +             fgrad(k) = fgrad(k) + trule*(dvdx/hx+dvdy/hy)
            end if
   70    continue
   80 continue

c     Computation of the linear part of the function and
c     corresponding components of the gradient.

      do 110 i = 1, nx
         temp = sin(dble(i)*hx)
         if (feval) then
            do 90 j = 1, ny
               k = nx*(j-1) + i
               flin = flin + temp*x(k)
   90       continue
         end if
         if (geval) then
            do 100 j = 1, ny
               k = nx*(j-1) + i
               fgrad(k) = fgrad(k) - ehxhy*temp
  100       continue
         end if
  110 continue

c     Finish off the function.

      if (feval) f = p5*fquad - ehxhy*flin

      end

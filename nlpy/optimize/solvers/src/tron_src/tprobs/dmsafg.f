      subroutine dmsafg(nx,ny,x,f,fgrad,task,bottom,top,left,right)
      character*60 task
      integer nx, ny
      double precision f
      double precision x(nx*ny), fgrad(nx*ny), bottom(nx+2), top(nx+2),
     +                 left(ny+2), right(ny+2)
c     **********
c
c     Subroutine dmsafg
c
c     This subroutine computes the function and gradient of the
c     minimal surface area problem.
c
c     The subroutine statement is
c
c       subroutine dmsafg(nx,ny,x,f,fgrad,task,bottom,top,left,right)
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
c
c         On exit task is unchanged.
c
c       bottom is a double precision array of dimension nx + 2.
c         On entry bottom must contain boundary data beginning
c            with the lower left corner of the domain.
c         On exit bottom is unchanged.
c
c       top is a double precision array of dimension nx + 2.
c         On entry top must contain boundary data beginning with
c            the upper left corner of the domain.
c         On exit top is unchanged.
c
c       left is a double precision array of dimension ny + 2.
c         On entry left must contain boundary data beginning with
c            the lower left corner of the domain.
c         On exit left is unchanged.
c
c       right is a double precision array of dimension ny + 2.
c         On entry right must contain boundary data beginning with
c            the lower right corner of the domain.
c         On exit right is unchanged.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory and University of Minnesota.
c     Brett M. Averick.
c
c     **********
      double precision one, p5, two, zero
      parameter (zero=0.0d0,p5=0.5d0,one=1.0d0,two=2.0d0)

      logical feval, geval
      integer i, j, k
      double precision alphaj, area, betai, dvdx, dvdy, fl, fu, hx, hy,
     +                 v, vb, vl, vr, vt, xline, yline

c     Initialize.

      hx = one/dble(nx+1)
      hy = one/dble(ny+1)
      area = p5*hx*hy

c     Compute the standard starting point if task = 'XS'.

      if (task(1:2) .eq. 'XS') then
         do 20 j = 1, ny
            alphaj = dble(j)*hy
            do 10 i = 1, nx
               k = nx*(j-1) + i
               betai = dble(i)*hx
               yline = alphaj*top(i+1) + (one-alphaj)*bottom(i+1)
               xline = betai*right(j+1) + (one-betai)*left(j+1)
               x(k) = (yline+xline)/two
   10       continue
   20    continue

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

c     Evaluate the function if task = 'F', the gradient if task = 'G',
c     or both if task = 'FG'.

      if (feval) f = zero
      if (geval) then
         do 30 k = 1, nx*ny
            fgrad(k) = zero
   30    continue
      end if

c     Computation of the function and gradient over the lower
c     triangular elements.

      do 50 j = 0, ny
         do 40 i = 0, nx
            k = nx*(j-1) + i
            if (i .ge. 1 .and. j .ge. 1) then
               v = x(k)
            else
               if (j .eq. 0) v = bottom(i+1)
               if (i .eq. 0) v = left(j+1)
            end if
            if (i .lt. nx .and. j .gt. 0) then
               vr = x(k+1)
            else
               if (i .eq. nx) vr = right(j+1)
               if (j .eq. 0) vr = bottom(i+2)
            end if
            if (i .gt. 0 .and. j .lt. ny) then
               vt = x(k+nx)
            else
               if (i .eq. 0) vt = left(j+2)
               if (j .eq. ny) vt = top(i+1)
            end if
            dvdx = (vr-v)/hx
            dvdy = (vt-v)/hy
            fl = sqrt(one+dvdx**2+dvdy**2)
            if (feval) f = f + fl
            if (geval) then
               if (i .ge. 1 .and. j .ge. 1)
     +             fgrad(k) = fgrad(k) - (dvdx/hx+dvdy/hy)/fl
               if (i .lt. nx .and. j .gt. 0)
     +             fgrad(k+1) = fgrad(k+1) + (dvdx/hx)/fl
               if (i .gt. 0 .and. j .lt. ny)
     +             fgrad(k+nx) = fgrad(k+nx) + (dvdy/hy)/fl
            end if
   40    continue
   50 continue

c     Computation of the function and the gradient over the upper
c     triangular elements.

      do 70 j = 1, ny + 1
         do 60 i = 1, nx + 1
            k = nx*(j-1) + i
            if (i .le. nx .and. j .gt. 1) then
               vb = x(k-nx)
            else
               if (j .eq. 1) vb = bottom(i+1)
               if (i .eq. nx+1) vb = right(j)
            end if
            if (i .gt. 1 .and. j .le. ny) then
               vl = x(k-1)
            else
               if (j .eq. ny+1) vl = top(i)
               if (i .eq. 1) vl = left(j+1)
            end if
            if (i .le. nx .and. j .le. ny) then
               v = x(k)
            else
               if (i .eq. nx+1) v = right(j+1)
               if (j .eq. ny+1) v = top(i+1)
            end if
            dvdx = (v-vl)/hx
            dvdy = (v-vb)/hy
            fu = sqrt(one+dvdx**2+dvdy**2)
            if (feval) f = f + fu
            if (geval) then
               if (i .le. nx .and. j .gt. 1)
     +             fgrad(k-nx) = fgrad(k-nx) - (dvdy/hy)/fu
               if (i .gt. 1 .and. j .le. ny)
     +             fgrad(k-1) = fgrad(k-1) - (dvdx/hx)/fu
               if (i .le. nx .and. j .le. ny)
     +             fgrad(k) = fgrad(k) + (dvdx/hx+dvdy/hy)/fu
            end if
   60    continue
   70 continue

c     Scale the function and the gradient.

      if (feval) f = area*f
      if (geval) then
         do 80 k = 1, nx*ny
            fgrad(k) = area*fgrad(k)
   80    continue
      end if

      end

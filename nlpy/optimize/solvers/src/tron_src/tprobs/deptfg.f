      subroutine deptfg(nx,ny,x,f,fgrad,task,c)
      character*60 task
      integer nx, ny
      double precision f, c
      double precision x(nx*ny), fgrad(nx*ny)
c     **********
c
c     Subroutine deptfg
c
c     This subroutine computes the function and gradient of the
c     elastic-plastic torsion problem.
c
c     The subroutine statement is
c
c       subroutine deptfg(nx,ny,x,f,fgrad,task,c)
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
c             'XU'    Set x to the upper bound xu.
c
c         On exit task is unchanged.
c
c       c is a double precision variable.
c         On entry c is the angle of twist per unit length.
c         On exit c is unchanged.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory and University of Minnesota.
c     Brett M. Averick and Jorge J. More'.
c
c     **********
      double precision one, p5, three, zero
      parameter (zero=0.0d0,p5=0.5d0,one=1.0d0,three=3.0d0)

      logical feval, geval
      integer i, j, k
      double precision area, cdiv3, dvdx, dvdy, flin, fquad, hx, hy,
     +                 temp, temp1, v, vb, vl, vr, vt

      hx = one/dble(nx+1)
      hy = one/dble(ny+1)
      area = p5*hx*hy
      cdiv3 = c/three

c     Compute a lower bound for x if task = 'XL' or an upper bound if
c     task = 'XU'.

      if (task(1:2) .eq. 'XL' .or. task(1:2) .eq. 'XU') then
         if (task(1:2) .eq. 'XL') temp1 = -one
         if (task(1:2) .eq. 'XU') temp1 = one
         do 20 j = 1, ny
            temp = dble(min(j,ny-j+1))*hy
            do 10 i = 1, nx
               k = nx*(j-1) + i
               x(k) = sign(min(dble(min(i,nx-i+1))*hx,temp),temp1)
   10       continue
   20    continue

         return

      end if

c     Compute the standard starting point if task = 'XS'.

      if (task(1:2) .eq. 'XS') then
         do 40 j = 1, ny
            temp = dble(min(j,ny-j+1))*hy
            do 30 i = 1, nx
               k = nx*(j-1) + i
               x(k) = min(dble(min(i,nx-i+1))*hx,temp)
   30       continue
   40    continue

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

      if (feval) then
         fquad = zero
         flin = zero
      end if
      if (geval) then
         do 50 k = 1, nx*ny
            fgrad(k) = zero
   50    continue
      end if

c     Computation of the function and the gradient over the lower
c     triangular elements.

      do 70 j = 0, ny
         do 60 i = 0, nx
            k = nx*(j-1) + i
            v = zero
            vr = zero
            vt = zero
            if (i .ge. 1 .and. j .ge. 1) v = x(k)
            if (i .lt. nx .and. j .gt. 0) vr = x(k+1)
            if (i .gt. 0 .and. j .lt. ny) vt = x(k+nx)
            dvdx = (vr-v)/hx
            dvdy = (vt-v)/hy
            if (feval) then
               fquad = fquad + dvdx**2 + dvdy**2
               flin = flin - cdiv3*(v+vr+vt)
            end if
            if (geval) then
               if (i .ne. 0 .and. j .ne. 0)
     +             fgrad(k) = fgrad(k) - dvdx/hx - dvdy/hy - cdiv3
               if (i .ne. nx .and. j .ne. 0)
     +             fgrad(k+1) = fgrad(k+1) + dvdx/hx - cdiv3
               if (i .ne. 0 .and. j .ne. ny)
     +             fgrad(k+nx) = fgrad(k+nx) + dvdy/hy - cdiv3
            end if
   60    continue
   70 continue

c     Computation of the function and the gradient over the upper
c     triangular elements.

      do 90 j = 1, ny + 1
         do 80 i = 1, nx + 1
            k = nx*(j-1) + i
            vb = zero
            vl = zero
            v = zero
            if (i .le. nx .and. j .gt. 1) vb = x(k-nx)
            if (i .gt. 1 .and. j .le. ny) vl = x(k-1)
            if (i .le. nx .and. j .le. ny) v = x(k)
            dvdx = (v-vl)/hx
            dvdy = (v-vb)/hy
            if (feval) then
               fquad = fquad + dvdx**2 + dvdy**2
               flin = flin - cdiv3*(vb+vl+v)
            end if
            if (geval) then
               if (i .ne. nx+1 .and. j .ne. 1)
     +             fgrad(k-nx) = fgrad(k-nx) - dvdy/hy - cdiv3
               if (i .ne. 1 .and. j .ne. ny+1)
     +             fgrad(k-1) = fgrad(k-1) - dvdx/hx - cdiv3
               if (i .ne. nx+1 .and. j .ne. ny+1)
     +             fgrad(k) = fgrad(k) + dvdx/hx + dvdy/hy - cdiv3
            end if
   80    continue
   90 continue

c     Scale the result.

      if (feval) f = area*(p5*fquad+flin)
      if (geval) then
         do 100 k = 1, nx*ny
            fgrad(k) = area*fgrad(k)
  100    continue
      end if

      end

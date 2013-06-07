      subroutine dodcfg(nx,ny,x,f,fgrad,task,lambda)
      character*60 task
      integer nx, ny
      double precision f, lambda
      double precision x(nx*ny), fgrad(nx*ny)
c     **********
c
c     Subroutine dodcfg
c
c     This subroutine computes the function and gradient of the
c     optimal design with composite materials problem.
c
c     The subroutine statement is
c
c       subroutine dodcfg(nx,ny,x,f,fgrad,task,lambda)
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
c       lambda is a double precision variable.
c         On entry lambda is the Lagrange multiplier.
c         On exit lambda is unchanged.
c
c     Subprograms called
c
c       MINPACK-supplied   ...   dodcps
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory and University of Minnesota.
c     Brett M. Averick.
c
c     **********
      double precision mu1, mu2, one, p5, two, zero
      parameter (zero=0.0d0,p5=0.5d0,one=1.0d0,two=2.0d0)
      parameter (mu1=one,mu2=two)

      logical feval, geval
      integer i, j, k
      double precision area, dpsi, dpsip, dvdx, dvdy, gradv, hx, hxhy,
     +                 hy, temp, t1, t2, v, vb, vl, vr, vt

      external dodcps

c     Initialization.

      hx = one/dble(nx+1)
      hy = one/dble(ny+1)
      hxhy = hx*hy
      area = p5*hxhy

c     Compute the break points.

      t1 = sqrt(two*lambda*mu1/mu2)
      t2 = sqrt(two*lambda*mu2/mu1)

c     Compute the standard starting point if task = 'XS'.

      if (task(1:2) .eq. 'XS') then
         do 20 j = 1, ny
            temp = dble(min(j,ny-j+1))*hy
            do 10 i = 1, nx
               k = nx*(j-1) + i
               x(k) = -(min(dble(min(i,nx-i+1))*hx,temp))**2
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

c     Computation of the function and the gradient over the lower
c     triangular elements.

      do 50 j = 0, ny
         do 40 i = 0, nx
            k = nx*(j-1) + i
            v = zero
            vr = zero
            vt = zero
            if (j .ge. 1 .and. i .ge. 1) v = x(k)
            if (i .lt. nx .and. j .gt. 0) vr = x(k+1)
            if (i .gt. 0 .and. j .lt. ny) vt = x(k+nx)
            dvdx = (vr-v)/hx
            dvdy = (vt-v)/hy
            gradv = dvdx**2 + dvdy**2
            if (feval) then
               call dodcps(gradv,mu1,mu2,t1,t2,dpsi,0,lambda)
               f = f + dpsi
            end if
            if (geval) then
               call dodcps(gradv,mu1,mu2,t1,t2,dpsip,1,lambda)
               if (i .ge. 1 .and. j .ge. 1)
     +             fgrad(k) = fgrad(k) - two*(dvdx/hx+dvdy/hy)*dpsip
               if (i .lt. nx .and. j .gt. 0)
     +             fgrad(k+1) = fgrad(k+1) + two*(dvdx/hx)*dpsip
               if (i .gt. 0 .and. j .lt. ny)
     +             fgrad(k+nx) = fgrad(k+nx) + two*(dvdy/hy)*dpsip
            end if
   40    continue
   50 continue

c     Computation of the function and the gradient over the upper
c     triangular elements.

      do 70 j = 1, ny + 1
         do 60 i = 1, nx + 1
            k = nx*(j-1) + i
            vb = zero
            vl = zero
            v = zero
            if (i .le. nx .and. j .gt. 1) vb = x(k-nx)
            if (i .gt. 1 .and. j .le. ny) vl = x(k-1)
            if (i .le. nx .and. j .le. ny) v = x(k)
            dvdx = (v-vl)/hx
            dvdy = (v-vb)/hy
            gradv = dvdx**2 + dvdy**2
            if (feval) then
               call dodcps(gradv,mu1,mu2,t1,t2,dpsi,0,lambda)
               f = f + dpsi
            end if
            if (geval) then
               call dodcps(gradv,mu1,mu2,t1,t2,dpsip,1,lambda)
               if (i .le. nx .and. j .gt. 1)
     +             fgrad(k-nx) = fgrad(k-nx) - two*(dvdy/hy)*dpsip
               if (i .gt. 1 .and. j .le. ny)
     +             fgrad(k-1) = fgrad(k-1) - two*(dvdx/hx)*dpsip
               if (i .le. nx .and. j .le. ny)
     +             fgrad(k) = fgrad(k) + two*(dvdx/hx+dvdy/hy)*dpsip
            end if
   60    continue
   70 continue

c     Scale the function.

      if (feval) f = area*f

c     Integrate v over the domain.

      if (feval) then
         temp = zero
         do 80 k = 1, nx*ny
            temp = temp + x(k)
   80    continue
         f = f + hxhy*temp
      end if
      if (geval) then
         do 90 k = 1, nx*ny
            fgrad(k) = area*fgrad(k) + hxhy
   90    continue
      end if

      end

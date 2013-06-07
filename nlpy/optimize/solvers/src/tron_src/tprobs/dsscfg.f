      subroutine dsscfg(nx,ny,x,f,fgrad,task,lambda)
      character*60 task
      integer nx, ny
      double precision f, lambda
      double precision x(nx*ny), fgrad(nx*ny)
c     **********
c
c     Subroutine dsscfg
c
c     This subroutine computes the function and gradient of the
c     steady state combustion problem.
c
c     The subroutine statement is
c
c       subroutine dsscfg(nx,ny,x,f,fgrad,task,lambda)
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
c            'F'      Evaluate the function at x.
c            'G'      Evaluate the gradient at x.
c            'FG'     Evaluate the function and the gradient at x.
c            'XS'     Set x to the standard starting point xs.
c            'XL'     Set x to the lower bound xl.
c            'XU'     Set x to the upper bound xu.
c
c         On exit task is unchanged.
c
c       lambda is a double precision variable.
c         On entry lambda is a nonnegative Frank-Kamenetski parameter.
c         On exit lambda is unchanged.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory and University of Minnesota.
c     Brett M. Averick.
c
c     **********
      double precision one, p5, three, zero
      parameter (zero=0.0d0,p5=0.5d0,one=1.0d0,three=3.0d0)

      logical feval, geval
      integer i, j, k
      double precision area, dvdx, dvdy, expv, expvb, expvl, expvr,
     +                 expvt, fexp, fquad, hx, hy, v, vb, vl, vr, vt,
     +                 temp, temp1

c     Initialization.

      hx = one/dble(nx+1)
      hy = one/dble(ny+1)
      area = p5*hx*hy

c     Compute the standard starting point if task = 'XS'.

      if (task(1:2) .eq. 'XS') then

         temp1 = lambda/(lambda+one)
         do 20 j = 1, ny
            temp = dble(min(j,ny-j+1))*hy
            do 10 i = 1, nx
               k = nx*(j-1) + i
               x(k) = temp1*sqrt(min(dble(min(i,nx-i+1))*hx,temp))
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

c     Compute the function if task = 'F', the gradient if task = 'G', or
c     both if task = 'FG'.

      if (feval) then
         fquad = zero
         fexp = zero
      end if
      if (geval) then
         do 30 k = 1, nx*ny
            fgrad(k) = zero
   30    continue
      end if

c     Computation of the function and the gradient over the lower
c     triangular elements.  The trapezoidal rule is used to estimate
c     the integral of the exponential term.

      do 50 j = 0, ny
         do 40 i = 0, nx
            k = nx*(j-1) + i
            v = zero
            vr = zero
            vt = zero
            if (i .ne. 0 .and. j .ne. 0) v = x(k)
            if (i .ne. nx .and. j .ne. 0) vr = x(k+1)
            if (i .ne. 0 .and. j .ne. ny) vt = x(k+nx)
            dvdx = (vr-v)/hx
            dvdy = (vt-v)/hy
            expv = exp(v)
            expvr = exp(vr)
            expvt = exp(vt)
            if (feval) then
               fquad = fquad + dvdx**2 + dvdy**2
               fexp = fexp - lambda*(expv+expvr+expvt)/three
            end if
            if (geval) then
               if (i .ne. 0 .and. j .ne. 0) fgrad(k) = fgrad(k) -
     +             dvdx/hx - dvdy/hy - lambda*expv/three
               if (i .ne. nx .and. j .ne. 0) fgrad(k+1) = fgrad(k+1) +
     +             dvdx/hx - lambda*expvr/three
               if (i .ne. 0 .and. j .ne. ny) fgrad(k+nx) = fgrad(k+nx) +
     +             dvdy/hy - lambda*expvt/three
            end if
   40    continue
   50 continue

c     Computation of the function and the gradient over the upper
c     triangular elements.  The trapezoidal rule is used to estimate
c     the integral of the exponential term.

      do 70 j = 1, ny + 1
         do 60 i = 1, nx + 1
            k = nx*(j-1) + i
            vb = zero
            vl = zero
            v = zero
            if (i .ne. nx+1 .and. j .ne. 1) vb = x(k-nx)
            if (i .ne. 1 .and. j .ne. ny+1) vl = x(k-1)
            if (i .ne. nx+1 .and. j .ne. ny+1) v = x(k)
            dvdx = (v-vl)/hx
            dvdy = (v-vb)/hy
            expvb = exp(vb)
            expvl = exp(vl)
            expv = exp(v)
            if (feval) then
               fquad = fquad + dvdx**2 + dvdy**2
               fexp = fexp - lambda*(expvb+expvl+expv)/three
            end if
            if (geval) then
               if (i .ne. nx+1 .and. j .ne.
     +             1) fgrad(k-nx) = fgrad(k-nx) - dvdy/hy -
     +             lambda*expvb/three
               if (i .ne. 1 .and. j .ne. ny+1) fgrad(k-1) = fgrad(k-1) -
     +             dvdx/hx - lambda*expvl/three
               if (i .ne. nx+1 .and. j .ne. ny+1) fgrad(k) = fgrad(k) +
     +             dvdx/hx + dvdy/hy - lambda*expv/three
            end if
   60    continue
   70 continue

c     Scale the result.

      if (feval) f = area*(p5*fquad+fexp)
      if (geval) then
         do 80 k = 1, nx*ny
            fgrad(k) = area*fgrad(k)
   80    continue
      end if

      end

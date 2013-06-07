      subroutine dminxb(n,nx,ny,xl,xu,prob) 
      character*6 prob
      integer n, nx, ny
      double precision xl(*), xu(*)
c     *********
c
c     Subroutine dminxb
c
c     This subroutine computes the lower and upper bounds for
c     the minimization problem from the MINPACK-2 test problem 
c     collection specified by the character variable prob.
c
c     The subroutine statement is
c
c       dminxb(n,nx,ny,xl,xu,prob)
c
c     where
c
c       n is an integer variable.
c         On entry n is the number of variables.
c         On exit n is unchanged.
c
c       nx is an integer variable.
c         On entry nx is the number of grid points in the first
c            coordinate direction.
c         On exit nx is unchanged.
c
c       ny is an integer variable.
c         On entry ny is the number of grid points in the second
c            coordinate direction. If the problem is formulated in
c            one spatial dimension, ny = 1.
c         On exit ny is unchanged.
c
c       xl is a double precision array of dimension n.
c         On entry xl need not be specified.
c         On exit xl is the vector of lower bounds.
c
c       xu is a double precision array of dimension n.
c         On entry xu need not be specified.
c         On exit xu is the vector of upper bounds.
c
c       prob is a character*6 variable.
c         On entry prob specifies the problem.
c         On exit prob is unchanged.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Brett M. Averick and Jorge J. More'.
c
c     **********
      double precision zero, one
      parameter(zero=0.0d0,one=1.0d0)
      double precision p01, p1, p5
      parameter(p01=1.0d-2,p1=1.0d-1,p5=5.0d-1)
      double precision oned2, xbmax
      parameter(oned2=1.0d2,xbmax=1.0d20)

      integer i, j, k
      double precision tmp
      double precision hx, hy
      double precision height
      double precision plx, ply, xlc, xuc

c     Select a problem.       

      if (prob(1:4) .eq. 'DEPT') then
         hx = one/dble(nx+1)
         hy = one/dble(ny+1)
         do j = 1, ny
            tmp = dble(min(j,ny-j+1))*hy
            do i = 1, nx
               k = nx*(j-1) + i
               xu(k) = min(dble(min(i,nx-i+1))*hx,tmp)
               xl(k) = -xu(k)
            end do
          end do
      else if (prob(1:4) .eq. 'DMSA') then
         if (prob(1:6) .eq. 'DMSA_A') height = zero
         if (prob(1:6) .eq. 'DMSA_B') height = 2*one
         if (prob(1:6) .eq. 'DMSA_C') height = 4*one
         hx = one/dble(nx+1)
         hy = one/dble(ny+1)
         xuc = max(2*height,one)
         do j = 1, ny
            do i = 1, nx
               k = nx*(j-1) + i
               xl(k) = height - 10*(abs(i*hx-p5) + abs(j*hy-p5))
               xu(k) = xuc
            end do
          end do
      else if (prob(1:4) .eq. 'DMSO') then
         hx = one/dble(nx+1)
         hy = one/dble(ny+1)
         plx = 25*p01
         ply = 25*p01
         if (prob(1:6) .eq. 'DMSO_A') xlc = zero
         if (prob(1:6) .eq. 'DMSO_B') xlc = 5*p1
         if (prob(1:6) .eq. 'DMSO_C') xlc = one
         xuc = 2*one
         do j = 1, ny
            do i = 1, nx
               k = nx*(j-1) + i
               xl(k) = zero
               if (abs(i*hx-p5) .le. plx .and. abs(j*hy-p5) .le. ply) 
     +            xl(k) = xlc
               xu(k) = xuc
            end do
          end do
      else if (prob(1:4) .eq. 'DPJB') then
         do i = 1, n
            xl(i) = zero
            xu(i) = oned2
         end do
      else if (prob(1:4) .eq. 'DSSC') then
         if (prob(1:6) .eq. 'DSSC_A') then
            xlc = zero
            xuc = 50*p01
         end if
         if (prob(1:6) .eq. 'DSSC_B') then
            xlc = zero
            xuc = 25*p01
         end if
         if (prob(1:6) .eq. 'DSSC_C') then
            xlc = zero
            xuc = 10*p01
         end if
         do i = 1, n
            xl(i) = xlc
            xu(i) = xuc
         end do
      else
         do i = 1, n
            xl(i) = -xbmax
            xu(i) =  xbmax
         end do
      end if

      end

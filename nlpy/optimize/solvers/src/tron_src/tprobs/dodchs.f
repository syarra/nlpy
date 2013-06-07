      subroutine dodchs(nx,ny,x,s,y,lambda)
      integer nx, ny
      double precision lambda
      double precision x(nx*ny), s(nx*ny), y(nx*ny)
c     **********
c
c     Subroutine dodchs
c
c     This subroutine computes the productc f''(x)*s = y, where
c     f''(x) is the Hessian matrix for the optimal design with
c     composites problem evaluated at x.
c
c     The subroutine statement is
c
c       subroutine dodchs(nx,ny,x,s,y,lambda)
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
c         On entry x specifies the vector x.
c         On exit x is unchanged.
c
c       s is a double precision array of dimension nx*ny.
c         On entry s contains the vector s.
c         On exit s is unchanged.
c
c       y is a double precision array of dimension nx*ny.
c         On entry out need not be specified.
c         On exit y contains f''(x)*s.
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
      double precision four, one, p5, two, zero
      parameter (zero=0.0d0,p5=0.5d0,one=1.0d0,two=2.0d0,four=4.0d0)
      double precision mu1, mu2
      parameter (mu1=1.0d0,mu2=2.0d0)

      integer i, j, k
      double precision area, dpsip, dpsipp, dvdx, dvdxhx, dvdy, dvdyhy,
     +                 dzdx, dzdxhx, dzdy, dzdyhy, gradv, hx, hy, t1,
     +                 t2, v, vb, vl, vr, vt, z, zb, zl, zr, zt

      external dodcps

      hx = one/dble(nx+1)
      hy = one/dble(ny+1)
      area = p5*hx*hy
      t1 = sqrt(two*lambda*mu1/mu2)
      t2 = sqrt(two*lambda*mu2/mu1)

      do 10 k = 1, nx*ny
         y(k) = zero
   10 continue

c     Computation of f''(x)*s over the lower triangular elements.

      do 30 j = 0, ny
         do 20 i = 0, nx
            k = nx*(j-1) + i
            v = zero
            vr = zero
            vt = zero
            z = zero
            zr = zero
            zt = zero
            if (i .ne. 0 .and. j .ne. 0) then
               v = x(k)
               z = s(k)
            end if
            if (i .ne. nx .and. j .ne. 0) then
               vr = x(k+1)
               zr = s(k+1)
            end if
            if (i .ne. 0 .and. j .ne. ny) then
               vt = x(k+nx)
               zt = s(k+nx)
            end if
            dvdx = (vr-v)/hx
            dvdy = (vt-v)/hy
            dzdx = (zr-z)/hx
            dzdy = (zt-z)/hy
            dvdxhx = dvdx/hx
            dvdyhy = dvdy/hy
            dzdxhx = dzdx/hx
            dzdyhy = dzdy/hy
            gradv = dvdx**2 + dvdy**2
            call dodcps(gradv,mu1,mu2,t1,t2,dpsip,1,lambda)
            call dodcps(gradv,mu1,mu2,t1,t2,dpsipp,2,lambda)
            if (i .ne. nx .and. j .ne. 0) y(k+1) = y(k+1) +
     +          four*dpsipp*dvdxhx*(dvdx*dzdx+dvdy*dzdy) +
     +          two*dpsip*dzdxhx
            if (i .ne. 0 .and. j .ne. ny) y(k+nx) = y(k+nx) +
     +          four*dpsipp*dvdyhy*(dvdx*dzdx+dvdy*dzdy) +
     +          two*dpsip*dzdyhy
            if (i .ne. 0 .and. j .ne. 0) y(k) = y(k) -
     +          four*dpsipp*(dvdxhx+dvdyhy)*(dvdx*dzdx+dvdy*dzdy) -
     +          two*dpsip*(dzdxhx+dzdyhy)
   20    continue
   30 continue

c     Computation of f''(x)*s over the upper triangular elements.

      do 50 j = 1, ny + 1
         do 40 i = 1, nx + 1
            k = nx*(j-1) + i
            vb = zero
            vl = zero
            v = zero
            zb = zero
            zl = zero
            z = zero
            if (i .ne. nx+1 .and. j .ne. 1) then
               vb = x(k-nx)
               zb = s(k-nx)
            end if
            if (i .ne. 1 .and. j .ne. ny+1) then
               vl = x(k-1)
               zl = s(k-1)
            end if
            if (i .ne. nx+1 .and. j .ne. ny+1) then
               v = x(k)
               z = s(k)
            end if
            dvdx = (v-vl)/hx
            dvdy = (v-vb)/hy
            dzdx = (z-zl)/hx
            dzdy = (z-zb)/hy
            dvdxhx = dvdx/hx
            dvdyhy = dvdy/hy
            dzdxhx = dzdx/hx
            dzdyhy = dzdy/hy
            gradv = dvdx**2 + dvdy**2
            call dodcps(gradv,mu1,mu2,t1,t2,dpsip,1,lambda)
            call dodcps(gradv,mu1,mu2,t1,t2,dpsipp,2,lambda)
            if (i .ne. nx+1 .and. j .ne. 1) y(k-nx) = y(k-nx) -
     +          four*dpsipp*dvdyhy*(dvdx*dzdx+dvdy*dzdy) -
     +          two*dpsip*dzdyhy
            if (i .ne. 1 .and. j .ne. ny+1) y(k-1) = y(k-1) -
     +          four*dpsipp*dvdxhx*(dvdx*dzdx+dvdy*dzdy) -
     +          two*dpsip*dzdxhx
            if (i .ne. nx+1 .and. j .ne. ny+1) y(k) = y(k) +
     +          four*dpsipp*(dvdxhx+dvdyhy)*(dvdx*dzdx+dvdy*dzdy) +
     +          two*dpsip*(dzdxhx+dzdyhy)
   40    continue
   50 continue

c     Scale the result by the area.

      do 60 k = 1, nx*ny
         y(k) = area*y(k)
   60 continue

      end

      subroutine dgl2co(gp,nx,ny,x,sx,ldsx,y,sy,ldsy,vpotx,svpotx,
     +                  ldsvpx,vpoty,svpoty,ldsvpy,yx,ldyx,yy,ldyy,
     +                  yvpotx,ldyvpx,yvpoty,ldyvpy,vornum)
      integer nx, ny, vornum, ldsx, ldsy, ldsvpx, ldsvpy, ldyx, ldyy,
     +        ldyvpx, ldyvpy
      double precision x(nx+1,ny+1), sx(ldsx,nx+1,ny+1), y(nx+1,ny+1),
     +                 sy(ldsy,nx+1,ny+1), vpotx(nx+1,ny+1),
     +                 svpotx(ldsvpx,nx+1,ny+1), vpoty(nx+1,ny+1),
     +                 svpoty(ldsvpy,nx+1,ny+1), yx(ldyx,nx,ny),
     +                 yy(ldyy,nx,ny), yvpotx(ldyvpx,nx,ny),
     +                 yvpoty(ldyvpy,nx,ny)
c     **********
c
c     Subroutine dgl2co
c
c     This subroutine computes the product f''(x)*s = y, where f''(x)
c     is the Hessian matrix for the Ginzburg-Landau (2-dimensional)
c     problem evaluted at x.
c
c     This subroutine was obtained by running dgl2fg.f through
c     ADIFOR, then through nag_polish, then edited for readibility.
c
c     The subroutine statement is
c
c       subroutine dgl2co(gp,nx,ny,x,sx,ldsx,y,sy,ldsy,vpotx,svpotx,
c                         ldsvpx,vpoty,svpoty,ldsvpy,yx,ldyx,yy,ldyy,
c                         yvpotx,ldyvpx,yvpoty,ldyvpy,vornum)
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Brett M. Averick.
c
c     **********
      integer gpmax
      parameter (gpmax=20)
      double precision five, four, one, three, two, zero
      parameter (zero=0.0d0,one=1.0d0,two=2.0d0,three=3.0d0,four=4.0d0,
     +          five=5.0d0)

      integer i, gp, gi, j
      double precision arg, bave, cfac, d1, d1bar, ffbar, d2, d2bar, d3,
     +                 d3bar, d4, d4bar, d5, d5bar, d6, d7, d7bar, d8,
     +                 d8bar, d9, d9bar, d10, d10bar, d11, d12, d13bar,
     +                 d14, d14bar, d15, d17, d17bar, d18, d18bar, d19,
     +                 d19bar, d20, d20bar, d21, d23, d23bar, d24,
     +                 d25bar, d26, d27, d29, ffield, fkinx1, fkinx2,
     +                 fkiny1, fkiny2, hx, hy, pi, sfac, sqn, sqrtv,
     +                 tkappa
      double precision gffld(gpmax), gfknx1(gpmax), gfknx2(gpmax),
     +                 gfkny1(gpmax), gfkny2(gpmax)

c     Initialize.

      tkappa = five
      hx = sqrt(vornum/two)*three/dble(nx)
      hy = sqrt(vornum/two)*three*sqrt(three)/dble(ny)
      sqn = dble(nx*ny)
      pi = four*atan(one)
      bave = two*pi*vornum*tkappa/(sqn*hx*hy)
      sqrtv = sqrt(dble(vornum))*pi

c     Enforce vortex constraint and boundary conditions.

c     Right face for order parameter and vector potential.

      do 50 j = 1, ny + 1
         arg = two*pi*vornum*(dble(j)-one)/dble(ny)
         d3bar = -sin(arg)
         d1bar = cos(arg)
         do 10 gi = 1, gp
            sx(gi,nx+1,j) = d1bar*sx(gi,1,j) + d3bar*sy(gi,1,j)
   10    continue
         x(nx+1,j) = x(1,j)*cos(arg) - y(1,j)*sin(arg)
         d3bar = cos(arg)
         d1bar = sin(arg)
         do 20 gi = 1, gp
            sy(gi,nx+1,j) = d1bar*sx(gi,1,j) + d3bar*sy(gi,1,j)
   20    continue
         y(nx+1,j) = x(1,j)*sin(arg) + y(1,j)*cos(arg)
         do 30 gi = 1, gp
            svpotx(gi,nx+1,j) = svpotx(gi,1,j)
   30    continue
         vpotx(nx+1,j) = vpotx(1,j)
         do 40 gi = 1, gp
            svpoty(gi,nx+1,j) = svpoty(gi,1,j)
   40    continue
         vpoty(nx+1,j) = vpoty(1,j) + two*pi*vornum/(dble(ny)*hy)
   50 continue

c     Top face for order parameter and vector potential.

      do 100 i = 1, nx + 1
         do 60 gi = 1, gp
            sx(gi,i,ny+1) = sx(gi,i,1)
   60    continue
         x(i,ny+1) = x(i,1)
         do 70 gi = 1, gp
            sy(gi,i,ny+1) = sy(gi,i,1)
   70    continue
         y(i,ny+1) = y(i,1)
         do 80 gi = 1, gp
            svpotx(gi,i,ny+1) = svpotx(gi,i,1)
   80    continue
         vpotx(i,ny+1) = vpotx(i,1)
         do 90 gi = 1, gp
            svpoty(gi,i,ny+1) = svpoty(gi,i,1)
   90    continue
         vpoty(i,ny+1) = vpoty(i,1)
  100 continue

      do 180 j = 1, ny
         do 170 i = 1, nx
            d1 = x(i,j)
            d2 = x(i,j)
            d3 = d2**2
            d5 = y(i,j)
            d6 = d5**2
            d7 = -one + d3 + d6
            if (d5 .ne. zero) then
               d5bar = 2*(d1*(d6/d5))
            else
               d5bar = zero
            end if
            if (d2 .ne. zero) then
               d2bar = 2*(d1*(d3/d2))
            else
               d2bar = zero
            end if
            do 110 gi = 1, gp
               yx(gi,i,j) = d7*sx(gi,i,j) + d2bar*sx(gi,i,j) +
     +                      d5bar*sy(gi,i,j)
  110       continue
            d1bar = (one/sqn)*two
            do 120 gi = 1, gp
               yx(gi,i,j) = d1bar*yx(gi,i,j)
  120       continue
            d1 = y(i,j)
            d2 = x(i,j)
            d3 = d2**2
            d5 = y(i,j)
            d6 = d5**2
            d7 = -one + d3 + d6
            if (d5 .ne. zero) then
               d5bar = 2*(d1*(d6/d5))
            else
               d5bar = zero
            end if
            if (d2 .ne. zero) then
               d2bar = 2*(d1*(d3/d2))
            else
               d2bar = zero
            end if
            do 130 gi = 1, gp
               yy(gi,i,j) = d7*sy(gi,i,j) + d2bar*sx(gi,i,j) +
     +                      d5bar*sy(gi,i,j)
  130       continue
            d1bar = (one/sqn)*two
            do 140 gi = 1, gp
               yy(gi,i,j) = d1bar*yy(gi,i,j)
  140       continue
            do 150 gi = 1, gp
               yvpotx(gi,i,j) = zero
  150       continue
            do 160 gi = 1, gp
               yvpoty(gi,i,j) = zero
  160       continue
  170    continue
  180 continue

c     Kinetic energy part, interior points

      do 420 i = 2, nx
         do 410 j = 2, ny
            d2 = x(i,j)
            d4 = hx*vpotx(i,j)
            d5 = cos(d4)
            d8 = y(i,j)
            d10 = hx*vpotx(i,j)
            d11 = sin(d10)
            d13bar = (two/(hx*hx*sqn))
            d8bar = d13bar*d11
            d9bar = cos(d10)*(d13bar*d8)*hx
            d2bar = -d13bar*d5
            d3bar = (-sin(d4)*(-d13bar*d2))*hx
            do 190 gi = 1, gp
               gfknx1(gi) = d13bar*sx(gi,i+1,j) + d2bar*sx(gi,i,j) +
     +                      d3bar*svpotx(gi,i,j) + d8bar*sy(gi,i,j) +
     +                      d9bar*svpotx(gi,i,j)
  190       continue
            fkinx1 = (two/(hx*hx*sqn))*(x(i+1,j)-d2*d5+d8*d11)
            d2 = y(i,j)
            d4 = hx*vpotx(i,j)
            d5 = cos(d4)
            d8 = x(i,j)
            d10 = hx*vpotx(i,j)
            d11 = sin(d10)
            d13bar = (two/(hx*hx*sqn))
            d8bar = -d13bar*d11
            d9bar = cos(d10)*(-d13bar*d8)*hx
            d2bar = -d13bar*d5
            d3bar = (-sin(d4)*(-d13bar*d2))*hx
            do 200 gi = 1, gp
               gfknx2(gi) = d13bar*sy(gi,i+1,j) + d2bar*sy(gi,i,j) +
     +                      d3bar*svpotx(gi,i,j) + d8bar*sx(gi,i,j) +
     +                      d9bar*svpotx(gi,i,j)
  200       continue
            fkinx2 = (two/(hx*hx*sqn))*(y(i+1,j)-d2*d5-d8*d11)
            d2 = x(i,j)
            d4 = hy*vpoty(i,j)
            d5 = cos(d4)
            d8 = y(i,j)
            d10 = hy*vpoty(i,j)
            d11 = sin(d10)
            d13bar = (two/(hy*hy*sqn))
            d8bar = d13bar*d11
            d9bar = cos(d10)*(d13bar*d8)*hy
            d2bar = -d13bar*d5
            d3bar = (-sin(d4)*(-d13bar*d2))*hy
            do 210 gi = 1, gp
               gfkny1(gi) = d13bar*sx(gi,i,j+1) + d2bar*sx(gi,i,j) +
     +                      d3bar*svpoty(gi,i,j) + d8bar*sy(gi,i,j) +
     +                      d9bar*svpoty(gi,i,j)
  210       continue
            fkiny1 = (two/(hy*hy*sqn))*(x(i,j+1)-d2*d5+d8*d11)
            d2 = y(i,j)
            d4 = hy*vpoty(i,j)
            d5 = cos(d4)
            d8 = x(i,j)
            d10 = hy*vpoty(i,j)
            d11 = sin(d10)
            d13bar = (two/(hy*hy*sqn))
            d8bar = -d13bar*d11
            d9bar = cos(d10)*(-d13bar*d8)*hy
            d2bar = -d13bar*d5
            d3bar = (-sin(d4)*(-d13bar*d2))*hy
            do 220 gi = 1, gp
               gfkny2(gi) = d13bar*sy(gi,i,j+1) + d2bar*sy(gi,i,j) +
     +                      d3bar*svpoty(gi,i,j) + d8bar*sx(gi,i,j) +
     +                      d9bar*svpoty(gi,i,j)
  220       continue
            fkiny2 = (two/(hy*hy*sqn))*(y(i,j+1)-d2*d5-d8*d11)
            d7bar = (one/hx)
            d3bar = (one/hy)
            do 230 gi = 1, gp
               gffld(gi) = d3bar*svpotx(gi,i,j) +
     +                     (-d3bar*svpotx(gi,i,j+1)) +
     +                     d7bar*svpoty(gi,i+1,j) +
     +                     (-d7bar*svpoty(gi,i,j))
  230       continue
            ffield = (vpotx(i,j)-vpotx(i,j+1))/hy +
     +               (vpoty(i+1,j)-vpoty(i,j))/hx
            ffbar = (two*(tkappa**2)/sqn)
            do 240 gi = 1, gp
               gffld(gi) = ffbar*gffld(gi)
  240       continue
            ffield = (two*(tkappa**2)/sqn)*ffield
            d3 = hx*vpotx(i,j)
            d5 = -cos(d3)
            d9 = hx*vpotx(i,j)
            d11 = -sin(d9)
            d15 = hy*vpoty(i,j)
            d17 = -cos(d15)
            d21 = hy*vpoty(i,j)
            d23 = -sin(d21)
            d20bar = cos(d21)*(-fkiny2)*hy
            d14bar = (-sin(d15)*(-fkiny1))*hy
            d8bar = cos(d9)*(-fkinx2)*hx
            d2bar = (-sin(d3)*(-fkinx1))*hx
            do 250 gi = 1, gp
               yx(gi,i,j) = d5*gfknx1(gi) + d11*gfknx2(gi) +
     +                      d17*gfkny1(gi) + d23*gfkny2(gi) +
     +                      yx(gi,i,j) + d2bar*svpotx(gi,i,j) +
     +                      d8bar*svpotx(gi,i,j) +
     +                      d14bar*svpoty(gi,i,j) +
     +                      d20bar*svpoty(gi,i,j)
  250       continue
            d3 = hx*vpotx(i,j)
            d4 = sin(d3)
            d8 = hx*vpotx(i,j)
            d10 = -cos(d8)
            d14 = hy*vpoty(i,j)
            d15 = sin(d14)
            d19 = hy*vpoty(i,j)
            d21 = -cos(d19)
            d18bar = (-sin(d19)*(-fkiny2))*hy
            d13bar = cos(d14)*fkiny1*hy
            d7bar = (-sin(d8)*(-fkinx2))*hx
            d2bar = cos(d3)*fkinx1*hx
            do 260 gi = 1, gp
               yy(gi,i,j) = d4*gfknx1(gi) + d10*gfknx2(gi) +
     +                      d15*gfkny1(gi) + d21*gfkny2(gi) +
     +                      yy(gi,i,j) + d2bar*svpotx(gi,i,j) +
     +                      d7bar*svpotx(gi,i,j) +
     +                      d13bar*svpoty(gi,i,j) +
     +                      d18bar*svpoty(gi,i,j)
  260       continue
            d3 = hx*x(i,j)
            d5 = hx*vpotx(i,j)
            d6 = sin(d5)
            d9 = hx*y(i,j)
            d11 = hx*vpotx(i,j)
            d12 = cos(d11)
            d14 = d3*d6 + d9*d12
            d18 = hx*y(i,j)
            d20 = hx*vpotx(i,j)
            d21 = sin(d20)
            d24 = hx*x(i,j)
            d26 = hx*vpotx(i,j)
            d27 = cos(d26)
            d29 = d18*d21 - d24*d27
            ffbar = (one/hy)
            d25bar = (-sin(d26)*(-fkinx2*d24))*hx
            d23bar = -fkinx2*d27*hx
            d19bar = cos(d20)*(fkinx2*d18)*hx
            d17bar = fkinx2*d21*hx
            d10bar = (-sin(d11)*(fkinx1*d9))*hx
            d8bar = fkinx1*d12*hx
            d4bar = cos(d5)*(fkinx1*d3)*hx
            d2bar = fkinx1*d6*hx
            do 270 gi = 1, gp
               yvpotx(gi,i,j) = d14*gfknx1(gi) + d29*gfknx2(gi) +
     +                          ffbar*gffld(gi) + yvpotx(gi,i,j) +
     +                          d2bar*sx(gi,i,j) +
     +                          d4bar*svpotx(gi,i,j) +
     +                          d8bar*sy(gi,i,j) +
     +                          d10bar*svpotx(gi,i,j) +
     +                          d17bar*sy(gi,i,j) +
     +                          d19bar*svpotx(gi,i,j) +
     +                          d23bar*sx(gi,i,j) +
     +                          d25bar*svpotx(gi,i,j)
  270       continue
            d3 = hy*x(i,j)
            d5 = hy*vpoty(i,j)
            d6 = sin(d5)
            d9 = hy*y(i,j)
            d11 = hy*vpoty(i,j)
            d12 = cos(d11)
            d14 = d3*d6 + d9*d12
            d18 = hy*y(i,j)
            d20 = hy*vpoty(i,j)
            d21 = sin(d20)
            d24 = hy*x(i,j)
            d26 = hy*vpoty(i,j)
            d27 = cos(d26)
            d29 = d18*d21 - d24*d27
            ffbar = -(one/hx)
            d25bar = (-sin(d26)*(-fkiny2*d24))*hy
            d23bar = -fkiny2*d27*hy
            d19bar = cos(d20)*(fkiny2*d18)*hy
            d17bar = fkiny2*d21*hy
            d10bar = (-sin(d11)*(fkiny1*d9))*hy
            d8bar = fkiny1*d12*hy
            d4bar = cos(d5)*(fkiny1*d3)*hy
            d2bar = fkiny1*d6*hy
            do 280 gi = 1, gp
               yvpoty(gi,i,j) = d14*gfkny1(gi) + d29*gfkny2(gi) +
     +                          ffbar*gffld(gi) + yvpoty(gi,i,j) +
     +                          d2bar*sx(gi,i,j) +
     +                          d4bar*svpoty(gi,i,j) +
     +                          d8bar*sy(gi,i,j) +
     +                          d10bar*svpoty(gi,i,j) +
     +                          d17bar*sy(gi,i,j) +
     +                          d19bar*svpoty(gi,i,j) +
     +                          d23bar*sx(gi,i,j) +
     +                          d25bar*svpoty(gi,i,j)
  280       continue
            d2 = x(i-1,j)
            d4 = hx*vpotx(i-1,j)
            d5 = cos(d4)
            d8 = y(i-1,j)
            d10 = hx*vpotx(i-1,j)
            d11 = sin(d10)
            d13bar = (two/(hx*hx*sqn))
            d8bar = d13bar*d11
            d9bar = cos(d10)*(d13bar*d8)*hx
            d2bar = -d13bar*d5
            d3bar = (-sin(d4)*(-d13bar*d2))*hx
            do 290 gi = 1, gp
               gfknx1(gi) = d13bar*sx(gi,i,j) + d2bar*sx(gi,i-1,j) +
     +                      d3bar*svpotx(gi,i-1,j) +
     +                      d8bar*sy(gi,i-1,j) + d9bar*svpotx(gi,i-1,j)
  290       continue
            fkinx1 = (two/(hx*hx*sqn))*(x(i,j)-d2*d5+d8*d11)
            d2 = y(i-1,j)
            d4 = hx*vpotx(i-1,j)
            d5 = cos(d4)
            d8 = x(i-1,j)
            d10 = hx*vpotx(i-1,j)
            d11 = sin(d10)
            d13bar = (two/(hx*hx*sqn))
            d8bar = -d13bar*d11
            d9bar = cos(d10)*(-d13bar*d8)*hx
            d2bar = -d13bar*d5
            d3bar = (-sin(d4)*(-d13bar*d2))*hx
            do 300 gi = 1, gp
               gfknx2(gi) = d13bar*sy(gi,i,j) + d2bar*sy(gi,i-1,j) +
     +                      d3bar*svpotx(gi,i-1,j) +
     +                      d8bar*sx(gi,i-1,j) + d9bar*svpotx(gi,i-1,j)
  300       continue
            fkinx2 = (two/(hx*hx*sqn))*(y(i,j)-d2*d5-d8*d11)
            d2 = x(i,j-1)
            d4 = hy*vpoty(i,j-1)
            d5 = cos(d4)
            d8 = y(i,j-1)
            d10 = hy*vpoty(i,j-1)
            d11 = sin(d10)
            d13bar = (two/(hy*hy*sqn))
            d8bar = d13bar*d11
            d9bar = cos(d10)*(d13bar*d8)*hy
            d2bar = -d13bar*d5
            d3bar = (-sin(d4)*(-d13bar*d2))*hy
            do 310 gi = 1, gp
               gfkny1(gi) = d13bar*sx(gi,i,j) + d2bar*sx(gi,i,j-1) +
     +                      d3bar*svpoty(gi,i,j-1) +
     +                      d8bar*sy(gi,i,j-1) + d9bar*svpoty(gi,i,j-1)
  310       continue
            fkiny1 = (two/(hy*hy*sqn))*(x(i,j)-d2*d5+d8*d11)
            d2 = y(i,j-1)
            d4 = hy*vpoty(i,j-1)
            d5 = cos(d4)
            d8 = x(i,j-1)
            d10 = hy*vpoty(i,j-1)
            d11 = sin(d10)
            d13bar = (two/(hy*hy*sqn))
            d8bar = -d13bar*d11
            d9bar = cos(d10)*(-d13bar*d8)*hy
            d2bar = -d13bar*d5
            d3bar = (-sin(d4)*(-d13bar*d2))*hy
            do 320 gi = 1, gp
               gfkny2(gi) = d13bar*sy(gi,i,j) + d2bar*sy(gi,i,j-1) +
     +                      d3bar*svpoty(gi,i,j-1) +
     +                      d8bar*sx(gi,i,j-1) + d9bar*svpoty(gi,i,j-1)
  320       continue
            fkiny2 = (two/(hy*hy*sqn))*(y(i,j)-d2*d5-d8*d11)
            do 330 gi = 1, gp
               yx(gi,i,j) = gfknx1(gi) + gfkny1(gi) + yx(gi,i,j)
  330       continue
            do 340 gi = 1, gp
               yy(gi,i,j) = gfknx2(gi) + gfkny2(gi) + yy(gi,i,j)
  340       continue
            d7bar = (one/hx)
            d3bar = (one/hy)
            do 350 gi = 1, gp
               gffld(gi) = d3bar*svpotx(gi,i,j-1) +
     +                     (-d3bar*svpotx(gi,i,j)) +
     +                     d7bar*svpoty(gi,i+1,j-1) +
     +                     (-d7bar*svpoty(gi,i,j-1))
  350       continue
            ffield = (vpotx(i,j-1)-vpotx(i,j))/hy +
     +               (vpoty(i+1,j-1)-vpoty(i,j-1))/hx
            ffbar = (two*(tkappa**2)/sqn)
            do 360 gi = 1, gp
               gffld(gi) = ffbar*gffld(gi)
  360       continue
            ffield = (two*(tkappa**2)/sqn)*ffield
            ffbar = -(one/hy)
            do 370 gi = 1, gp
               yvpotx(gi,i,j) = ffbar*gffld(gi) + yvpotx(gi,i,j)
  370       continue
            d7bar = (one/hx)
            d3bar = (one/hy)
            do 380 gi = 1, gp
               gffld(gi) = d3bar*svpotx(gi,i-1,j) +
     +                     (-d3bar*svpotx(gi,i-1,j+1)) +
     +                     d7bar*svpoty(gi,i,j) +
     +                     (-d7bar*svpoty(gi,i-1,j))
  380       continue
            ffield = (vpotx(i-1,j)-vpotx(i-1,j+1))/hy +
     +               (vpoty(i,j)-vpoty(i-1,j))/hx
            ffbar = (two*(tkappa**2)/sqn)
            do 390 gi = 1, gp
               gffld(gi) = ffbar*gffld(gi)
  390       continue
            ffield = (two*(tkappa**2)/sqn)*ffield
            ffbar = (one/hx)
            do 400 gi = 1, gp
               yvpoty(gi,i,j) = ffbar*gffld(gi) + yvpoty(gi,i,j)
  400       continue
  410    continue
  420 continue

c     Kinetic energy part, boundary points.

c     Bottom j = 1

      do 650 i = 2, nx
         d2 = x(i,1)
         d4 = hx*vpotx(i,1)
         d5 = cos(d4)
         d8 = y(i,1)
         d10 = hx*vpotx(i,1)
         d11 = sin(d10)
         d13bar = (two/(hx*hx*sqn))
         d8bar = d13bar*d11
         d9bar = cos(d10)*(d13bar*d8)*hx
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hx
         do 430 gi = 1, gp
            gfknx1(gi) = d13bar*sx(gi,i+1,1) + d2bar*sx(gi,i,1) +
     +                   d3bar*svpotx(gi,i,1) + d8bar*sy(gi,i,1) +
     +                   d9bar*svpotx(gi,i,1)
  430    continue
         fkinx1 = (two/(hx*hx*sqn))*(x(i+1,1)-d2*d5+d8*d11)
         d2 = y(i,1)
         d4 = hx*vpotx(i,1)
         d5 = cos(d4)
         d8 = x(i,1)
         d10 = hx*vpotx(i,1)
         d11 = sin(d10)
         d13bar = (two/(hx*hx*sqn))
         d8bar = -d13bar*d11
         d9bar = cos(d10)*(-d13bar*d8)*hx
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hx
         do 440 gi = 1, gp
            gfknx2(gi) = d13bar*sy(gi,i+1,1) + d2bar*sy(gi,i,1) +
     +                   d3bar*svpotx(gi,i,1) + d8bar*sx(gi,i,1) +
     +                   d9bar*svpotx(gi,i,1)
  440    continue
         fkinx2 = (two/(hx*hx*sqn))*(y(i+1,1)-d2*d5-d8*d11)
         d2 = x(i,1)
         d4 = hy*vpoty(i,1)
         d5 = cos(d4)
         d8 = y(i,1)
         d10 = hy*vpoty(i,1)
         d11 = sin(d10)
         d13bar = (two/(hy*hy*sqn))
         d8bar = d13bar*d11
         d9bar = cos(d10)*(d13bar*d8)*hy
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hy
         do 450 gi = 1, gp
            gfkny1(gi) = d13bar*sx(gi,i,2) + d2bar*sx(gi,i,1) +
     +                   d3bar*svpoty(gi,i,1) + d8bar*sy(gi,i,1) +
     +                   d9bar*svpoty(gi,i,1)
  450    continue
         fkiny1 = (two/(hy*hy*sqn))*(x(i,2)-d2*d5+d8*d11)
         d2 = y(i,1)
         d4 = hy*vpoty(i,1)
         d5 = cos(d4)
         d8 = x(i,1)
         d10 = hy*vpoty(i,1)
         d11 = sin(d10)
         d13bar = (two/(hy*hy*sqn))
         d8bar = -d13bar*d11
         d9bar = cos(d10)*(-d13bar*d8)*hy
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hy
         do 460 gi = 1, gp
            gfkny2(gi) = d13bar*sy(gi,i,2) + d2bar*sy(gi,i,1) +
     +                   d3bar*svpoty(gi,i,1) + d8bar*sx(gi,i,1) +
     +                   d9bar*svpoty(gi,i,1)
  460    continue
         fkiny2 = (two/(hy*hy*sqn))*(y(i,2)-d2*d5-d8*d11)
         d7bar = (one/hx)
         d3bar = (one/hy)
         do 470 gi = 1, gp
            gffld(gi) = d3bar*svpotx(gi,i,1) + (-d3bar*svpotx(gi,i,2)) +
     +                  d7bar*svpoty(gi,i+1,1) + (-d7bar*svpoty(gi,i,1))
  470    continue
         ffield = (vpotx(i,1)-vpotx(i,2))/hy +
     +            (vpoty(i+1,1)-vpoty(i,1))/hx
         ffbar = (two*(tkappa**2)/sqn)
         do 480 gi = 1, gp
            gffld(gi) = ffbar*gffld(gi)
  480    continue
         ffield = (two*(tkappa**2)/sqn)*ffield
         d3 = hx*vpotx(i,1)
         d5 = -cos(d3)
         d9 = hx*vpotx(i,1)
         d11 = -sin(d9)
         d15 = hy*vpoty(i,1)
         d17 = -cos(d15)
         d21 = hy*vpoty(i,1)
         d23 = -sin(d21)
         d20bar = cos(d21)*(-fkiny2)*hy
         d14bar = (-sin(d15)*(-fkiny1))*hy
         d8bar = cos(d9)*(-fkinx2)*hx
         d2bar = (-sin(d3)*(-fkinx1))*hx
         do 490 gi = 1, gp
            yx(gi,i,1) = d5*gfknx1(gi) + d11*gfknx2(gi) +
     +                   d17*gfkny1(gi) + d23*gfkny2(gi) + yx(gi,i,1) +
     +                   d2bar*svpotx(gi,i,1) + d8bar*svpotx(gi,i,1) +
     +                   d14bar*svpoty(gi,i,1) + d20bar*svpoty(gi,i,1)
  490    continue
         d3 = hx*vpotx(i,1)
         d4 = sin(d3)
         d8 = hx*vpotx(i,1)
         d10 = -cos(d8)
         d14 = hy*vpoty(i,1)
         d15 = sin(d14)
         d19 = hy*vpoty(i,1)
         d21 = -cos(d19)
         d18bar = (-sin(d19)*(-fkiny2))*hy
         d13bar = cos(d14)*fkiny1*hy
         d7bar = (-sin(d8)*(-fkinx2))*hx
         d2bar = cos(d3)*fkinx1*hx
         do 500 gi = 1, gp
            yy(gi,i,1) = d4*gfknx1(gi) + d10*gfknx2(gi) +
     +                   d15*gfkny1(gi) + d21*gfkny2(gi) + yy(gi,i,1) +
     +                   d2bar*svpotx(gi,i,1) + d7bar*svpotx(gi,i,1) +
     +                   d13bar*svpoty(gi,i,1) + d18bar*svpoty(gi,i,1)
  500    continue
         d3 = hx*x(i,1)
         d5 = hx*vpotx(i,1)
         d6 = sin(d5)
         d9 = hx*y(i,1)
         d11 = hx*vpotx(i,1)
         d12 = cos(d11)
         d14 = d3*d6 + d9*d12
         d18 = hx*y(i,1)
         d20 = hx*vpotx(i,1)
         d21 = sin(d20)
         d24 = hx*x(i,1)
         d26 = hx*vpotx(i,1)
         d27 = cos(d26)
         d29 = d18*d21 - d24*d27
         ffbar = (one/hy)
         d25bar = (-sin(d26)*(-fkinx2*d24))*hx
         d23bar = -fkinx2*d27*hx
         d19bar = cos(d20)*(fkinx2*d18)*hx
         d17bar = fkinx2*d21*hx
         d10bar = (-sin(d11)*(fkinx1*d9))*hx
         d8bar = fkinx1*d12*hx
         d4bar = cos(d5)*(fkinx1*d3)*hx
         d2bar = fkinx1*d6*hx
         do 510 gi = 1, gp
            yvpotx(gi,i,1) = d14*gfknx1(gi) + d29*gfknx2(gi) +
     +                       ffbar*gffld(gi) + yvpotx(gi,i,1) +
     +                       d2bar*sx(gi,i,1) + d4bar*svpotx(gi,i,1) +
     +                       d8bar*sy(gi,i,1) + d10bar*svpotx(gi,i,1) +
     +                       d17bar*sy(gi,i,1) + d19bar*svpotx(gi,i,1) +
     +                       d23bar*sx(gi,i,1) + d25bar*svpotx(gi,i,1)
  510    continue
         d3 = hy*x(i,1)
         d5 = hy*vpoty(i,1)
         d6 = sin(d5)
         d9 = hy*y(i,1)
         d11 = hy*vpoty(i,1)
         d12 = cos(d11)
         d14 = d3*d6 + d9*d12
         d18 = hy*y(i,1)
         d20 = hy*vpoty(i,1)
         d21 = sin(d20)
         d24 = hy*x(i,1)
         d26 = hy*vpoty(i,1)
         d27 = cos(d26)
         d29 = d18*d21 - d24*d27
         ffbar = -(one/hx)
         d25bar = (-sin(d26)*(-fkiny2*d24))*hy
         d23bar = -fkiny2*d27*hy
         d19bar = cos(d20)*(fkiny2*d18)*hy
         d17bar = fkiny2*d21*hy
         d10bar = (-sin(d11)*(fkiny1*d9))*hy
         d8bar = fkiny1*d12*hy
         d4bar = cos(d5)*(fkiny1*d3)*hy
         d2bar = fkiny1*d6*hy
         do 520 gi = 1, gp
            yvpoty(gi,i,1) = d14*gfkny1(gi) + d29*gfkny2(gi) +
     +                       ffbar*gffld(gi) + yvpoty(gi,i,1) +
     +                       d2bar*sx(gi,i,1) + d4bar*svpoty(gi,i,1) +
     +                       d8bar*sy(gi,i,1) + d10bar*svpoty(gi,i,1) +
     +                       d17bar*sy(gi,i,1) + d19bar*svpoty(gi,i,1) +
     +                       d23bar*sx(gi,i,1) + d25bar*svpoty(gi,i,1)
  520    continue
         d2 = x(i-1,1)
         d4 = hx*vpotx(i-1,1)
         d5 = cos(d4)
         d8 = y(i-1,1)
         d10 = hx*vpotx(i-1,1)
         d11 = sin(d10)
         d13bar = (two/(hx*hx*sqn))
         d8bar = d13bar*d11
         d9bar = cos(d10)*(d13bar*d8)*hx
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hx
         do 530 gi = 1, gp
            gfknx1(gi) = d13bar*sx(gi,i,1) + d2bar*sx(gi,i-1,1) +
     +                   d3bar*svpotx(gi,i-1,1) + d8bar*sy(gi,i-1,1) +
     +                   d9bar*svpotx(gi,i-1,1)
  530    continue
         fkinx1 = (two/(hx*hx*sqn))*(x(i,1)-d2*d5+d8*d11)
         d2 = y(i-1,1)
         d4 = hx*vpotx(i-1,1)
         d5 = cos(d4)
         d8 = x(i-1,1)
         d10 = hx*vpotx(i-1,1)
         d11 = sin(d10)
         d13bar = (two/(hx*hx*sqn))
         d8bar = -d13bar*d11
         d9bar = cos(d10)*(-d13bar*d8)*hx
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hx
         do 540 gi = 1, gp
            gfknx2(gi) = d13bar*sy(gi,i,1) + d2bar*sy(gi,i-1,1) +
     +                   d3bar*svpotx(gi,i-1,1) + d8bar*sx(gi,i-1,1) +
     +                   d9bar*svpotx(gi,i-1,1)
  540    continue
         fkinx2 = (two/(hx*hx*sqn))*(y(i,1)-d2*d5-d8*d11)
         d2 = x(i,ny)
         d4 = hy*vpoty(i,ny)
         d5 = cos(d4)
         d8 = y(i,ny)
         d10 = hy*vpoty(i,ny)
         d11 = sin(d10)
         d13bar = (two/(hy*hy*sqn))
         d8bar = d13bar*d11
         d9bar = cos(d10)*(d13bar*d8)*hy
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hy
         do 550 gi = 1, gp
            gfkny1(gi) = d13bar*sx(gi,i,ny+1) + d2bar*sx(gi,i,ny) +
     +                   d3bar*svpoty(gi,i,ny) + d8bar*sy(gi,i,ny) +
     +                   d9bar*svpoty(gi,i,ny)
  550    continue
         fkiny1 = (two/(hy*hy*sqn))*(x(i,ny+1)-d2*d5+d8*d11)
         d2 = y(i,ny)
         d4 = hy*vpoty(i,ny)
         d5 = cos(d4)
         d8 = x(i,ny)
         d10 = hy*vpoty(i,ny)
         d11 = sin(d10)
         d13bar = (two/(hy*hy*sqn))
         d8bar = -d13bar*d11
         d9bar = cos(d10)*(-d13bar*d8)*hy
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hy
         do 560 gi = 1, gp
            gfkny2(gi) = d13bar*sy(gi,i,ny+1) + d2bar*sy(gi,i,ny) +
     +                   d3bar*svpoty(gi,i,ny) + d8bar*sx(gi,i,ny) +
     +                   d9bar*svpoty(gi,i,ny)
  560    continue
         fkiny2 = (two/(hy*hy*sqn))*(y(i,ny+1)-d2*d5-d8*d11)
         do 570 gi = 1, gp
            yx(gi,i,1) = gfknx1(gi) + gfkny1(gi) + yx(gi,i,1)
  570    continue
         do 580 gi = 1, gp
            yy(gi,i,1) = gfknx2(gi) + gfkny2(gi) + yy(gi,i,1)
  580    continue
         d7bar = (one/hx)
         d3bar = (one/hy)
         do 590 gi = 1, gp
            gffld(gi) = d3bar*svpotx(gi,i,ny) +
     +                  (-d3bar*svpotx(gi,i,ny+1)) +
     +                  d7bar*svpoty(gi,i+1,ny) +
     +                  (-d7bar*svpoty(gi,i,ny))
  590    continue
         ffield = (vpotx(i,ny)-vpotx(i,ny+1))/hy +
     +            (vpoty(i+1,ny)-vpoty(i,ny))/hx
         ffbar = (two*(tkappa**2)/sqn)
         do 600 gi = 1, gp
            gffld(gi) = ffbar*gffld(gi)
  600    continue
         ffield = (two*(tkappa**2)/sqn)*ffield
         ffbar = -(one/hy)
         do 610 gi = 1, gp
            yvpotx(gi,i,1) = ffbar*gffld(gi) + yvpotx(gi,i,1)
  610    continue
         d7bar = (one/hx)
         d3bar = (one/hy)
         do 620 gi = 1, gp
            gffld(gi) = d3bar*svpotx(gi,i-1,1) +
     +                  (-d3bar*svpotx(gi,i-1,2)) +
     +                  d7bar*svpoty(gi,i,1) + (-d7bar*svpoty(gi,i-1,1))
  620    continue
         ffield = (vpotx(i-1,1)-vpotx(i-1,2))/hy +
     +            (vpoty(i,1)-vpoty(i-1,1))/hx
         ffbar = (two*(tkappa**2)/sqn)
         do 630 gi = 1, gp
            gffld(gi) = ffbar*gffld(gi)
  630    continue
         ffield = (two*(tkappa**2)/sqn)*ffield
         ffbar = (one/hx)
         do 640 gi = 1, gp
            yvpoty(gi,i,1) = ffbar*gffld(gi) + yvpoty(gi,i,1)
  640    continue
  650 continue

c     Left i = 1.

      do 880 j = 2, ny
         d2 = x(1,j)
         d4 = hx*vpotx(1,j)
         d5 = cos(d4)
         d8 = y(1,j)
         d10 = hx*vpotx(1,j)
         d11 = sin(d10)
         d13bar = (two/(hx*hx*sqn))
         d8bar = d13bar*d11
         d9bar = cos(d10)*(d13bar*d8)*hx
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hx
         do 660 gi = 1, gp
            gfknx1(gi) = d13bar*sx(gi,2,j) + d2bar*sx(gi,1,j) +
     +                   d3bar*svpotx(gi,1,j) + d8bar*sy(gi,1,j) +
     +                   d9bar*svpotx(gi,1,j)
  660    continue
         fkinx1 = (two/(hx*hx*sqn))*(x(2,j)-d2*d5+d8*d11)
         d2 = y(1,j)
         d4 = hx*vpotx(1,j)
         d5 = cos(d4)
         d8 = x(1,j)
         d10 = hx*vpotx(1,j)
         d11 = sin(d10)
         d13bar = (two/(hx*hx*sqn))
         d8bar = -d13bar*d11
         d9bar = cos(d10)*(-d13bar*d8)*hx
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hx
         do 670 gi = 1, gp
            gfknx2(gi) = d13bar*sy(gi,2,j) + d2bar*sy(gi,1,j) +
     +                   d3bar*svpotx(gi,1,j) + d8bar*sx(gi,1,j) +
     +                   d9bar*svpotx(gi,1,j)
  670    continue
         fkinx2 = (two/(hx*hx*sqn))*(y(2,j)-d2*d5-d8*d11)
         d2 = x(1,j)
         d4 = hy*vpoty(1,j)
         d5 = cos(d4)
         d8 = y(1,j)
         d10 = hy*vpoty(1,j)
         d11 = sin(d10)
         d13bar = (two/(hy*hy*sqn))
         d8bar = d13bar*d11
         d9bar = cos(d10)*(d13bar*d8)*hy
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hy
         do 680 gi = 1, gp
            gfkny1(gi) = d13bar*sx(gi,1,j+1) + d2bar*sx(gi,1,j) +
     +                   d3bar*svpoty(gi,1,j) + d8bar*sy(gi,1,j) +
     +                   d9bar*svpoty(gi,1,j)
  680    continue
         fkiny1 = (two/(hy*hy*sqn))*(x(1,j+1)-d2*d5+d8*d11)
         d2 = y(1,j)
         d4 = hy*vpoty(1,j)
         d5 = cos(d4)
         d8 = x(1,j)
         d10 = hy*vpoty(1,j)
         d11 = sin(d10)
         d13bar = (two/(hy*hy*sqn))
         d8bar = -d13bar*d11
         d9bar = cos(d10)*(-d13bar*d8)*hy
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hy
         do 690 gi = 1, gp
            gfkny2(gi) = d13bar*sy(gi,1,j+1) + d2bar*sy(gi,1,j) +
     +                   d3bar*svpoty(gi,1,j) + d8bar*sx(gi,1,j) +
     +                   d9bar*svpoty(gi,1,j)
  690    continue
         fkiny2 = (two/(hy*hy*sqn))*(y(1,j+1)-d2*d5-d8*d11)
         d7bar = (one/hx)
         d3bar = (one/hy)
         do 700 gi = 1, gp
            gffld(gi) = d3bar*svpotx(gi,1,j) +
     +                  (-d3bar*svpotx(gi,1,j+1)) +
     +                  d7bar*svpoty(gi,2,j) + (-d7bar*svpoty(gi,1,j))
  700    continue
         ffield = (vpotx(1,j)-vpotx(1,j+1))/hy +
     +            (vpoty(2,j)-vpoty(1,j))/hx
         ffbar = (two*(tkappa**2)/sqn)
         do 710 gi = 1, gp
            gffld(gi) = ffbar*gffld(gi)
  710    continue
         ffield = (two*(tkappa**2)/sqn)*ffield
         d3 = hx*vpotx(1,j)
         d5 = -cos(d3)
         d9 = hx*vpotx(1,j)
         d11 = -sin(d9)
         d15 = hy*vpoty(1,j)
         d17 = -cos(d15)
         d21 = hy*vpoty(1,j)
         d23 = -sin(d21)
         d20bar = cos(d21)*(-fkiny2)*hy
         d14bar = (-sin(d15)*(-fkiny1))*hy
         d8bar = cos(d9)*(-fkinx2)*hx
         d2bar = (-sin(d3)*(-fkinx1))*hx
         do 720 gi = 1, gp
            yx(gi,1,j) = d5*gfknx1(gi) + d11*gfknx2(gi) +
     +                   d17*gfkny1(gi) + d23*gfkny2(gi) + yx(gi,1,j) +
     +                   d2bar*svpotx(gi,1,j) + d8bar*svpotx(gi,1,j) +
     +                   d14bar*svpoty(gi,1,j) + d20bar*svpoty(gi,1,j)
  720    continue
         d3 = hx*vpotx(1,j)
         d4 = sin(d3)
         d8 = hx*vpotx(1,j)
         d10 = -cos(d8)
         d14 = hy*vpoty(1,j)
         d15 = sin(d14)
         d19 = hy*vpoty(1,j)
         d21 = -cos(d19)
         d18bar = (-sin(d19)*(-fkiny2))*hy
         d13bar = cos(d14)*fkiny1*hy
         d7bar = (-sin(d8)*(-fkinx2))*hx
         d2bar = cos(d3)*fkinx1*hx
         do 730 gi = 1, gp
            yy(gi,1,j) = d4*gfknx1(gi) + d10*gfknx2(gi) +
     +                   d15*gfkny1(gi) + d21*gfkny2(gi) + yy(gi,1,j) +
     +                   d2bar*svpotx(gi,1,j) + d7bar*svpotx(gi,1,j) +
     +                   d13bar*svpoty(gi,1,j) + d18bar*svpoty(gi,1,j)
  730    continue
         d3 = hx*x(1,j)
         d5 = hx*vpotx(1,j)
         d6 = sin(d5)
         d9 = hx*y(1,j)
         d11 = hx*vpotx(1,j)
         d12 = cos(d11)
         d14 = d3*d6 + d9*d12
         d18 = hx*y(1,j)
         d20 = hx*vpotx(1,j)
         d21 = sin(d20)
         d24 = hx*x(1,j)
         d26 = hx*vpotx(1,j)
         d27 = cos(d26)
         d29 = d18*d21 - d24*d27
         ffbar = (one/hy)
         d25bar = (-sin(d26)*(-fkinx2*d24))*hx
         d23bar = -fkinx2*d27*hx
         d19bar = cos(d20)*(fkinx2*d18)*hx
         d17bar = fkinx2*d21*hx
         d10bar = (-sin(d11)*(fkinx1*d9))*hx
         d8bar = fkinx1*d12*hx
         d4bar = cos(d5)*(fkinx1*d3)*hx
         d2bar = fkinx1*d6*hx
         do 740 gi = 1, gp
            yvpotx(gi,1,j) = d14*gfknx1(gi) + d29*gfknx2(gi) +
     +                       ffbar*gffld(gi) + yvpotx(gi,1,j) +
     +                       d2bar*sx(gi,1,j) + d4bar*svpotx(gi,1,j) +
     +                       d8bar*sy(gi,1,j) + d10bar*svpotx(gi,1,j) +
     +                       d17bar*sy(gi,1,j) + d19bar*svpotx(gi,1,j) +
     +                       d23bar*sx(gi,1,j) + d25bar*svpotx(gi,1,j)
  740    continue
         d3 = hy*x(1,j)
         d5 = hy*vpoty(1,j)
         d6 = sin(d5)
         d9 = hy*y(1,j)
         d11 = hy*vpoty(1,j)
         d12 = cos(d11)
         d14 = d3*d6 + d9*d12
         d18 = hy*y(1,j)
         d20 = hy*vpoty(1,j)
         d21 = sin(d20)
         d24 = hy*x(1,j)
         d26 = hy*vpoty(1,j)
         d27 = cos(d26)
         d29 = d18*d21 - d24*d27
         ffbar = -(one/hx)
         d25bar = (-sin(d26)*(-fkiny2*d24))*hy
         d23bar = -fkiny2*d27*hy
         d19bar = cos(d20)*(fkiny2*d18)*hy
         d17bar = fkiny2*d21*hy
         d10bar = (-sin(d11)*(fkiny1*d9))*hy
         d8bar = fkiny1*d12*hy
         d4bar = cos(d5)*(fkiny1*d3)*hy
         d2bar = fkiny1*d6*hy
         do 750 gi = 1, gp
            yvpoty(gi,1,j) = d14*gfkny1(gi) + d29*gfkny2(gi) +
     +                       ffbar*gffld(gi) + yvpoty(gi,1,j) +
     +                       d2bar*sx(gi,1,j) + d4bar*svpoty(gi,1,j) +
     +                       d8bar*sy(gi,1,j) + d10bar*svpoty(gi,1,j) +
     +                       d17bar*sy(gi,1,j) + d19bar*svpoty(gi,1,j) +
     +                       d23bar*sx(gi,1,j) + d25bar*svpoty(gi,1,j)
  750    continue
         d2 = x(nx,j)
         d4 = hx*vpotx(nx,j)
         d5 = cos(d4)
         d8 = y(nx,j)
         d10 = hx*vpotx(nx,j)
         d11 = sin(d10)
         d13bar = (two/(hx*hx*sqn))
         d8bar = d13bar*d11
         d9bar = cos(d10)*(d13bar*d8)*hx
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hx
         do 760 gi = 1, gp
            gfknx1(gi) = d13bar*sx(gi,nx+1,j) + d2bar*sx(gi,nx,j) +
     +                   d3bar*svpotx(gi,nx,j) + d8bar*sy(gi,nx,j) +
     +                   d9bar*svpotx(gi,nx,j)
  760    continue
         fkinx1 = (two/(hx*hx*sqn))*(x(nx+1,j)-d2*d5+d8*d11)
         d2 = y(nx,j)
         d4 = hx*vpotx(nx,j)
         d5 = cos(d4)
         d8 = x(nx,j)
         d10 = hx*vpotx(nx,j)
         d11 = sin(d10)
         d13bar = (two/(hx*hx*sqn))
         d8bar = -d13bar*d11
         d9bar = cos(d10)*(-d13bar*d8)*hx
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hx
         do 770 gi = 1, gp
            gfknx2(gi) = d13bar*sy(gi,nx+1,j) + d2bar*sy(gi,nx,j) +
     +                   d3bar*svpotx(gi,nx,j) + d8bar*sx(gi,nx,j) +
     +                   d9bar*svpotx(gi,nx,j)
  770    continue
         fkinx2 = (two/(hx*hx*sqn))*(y(nx+1,j)-d2*d5-d8*d11)
         d2 = x(1,j-1)
         d4 = hy*vpoty(1,j-1)
         d5 = cos(d4)
         d8 = y(1,j-1)
         d10 = hy*vpoty(1,j-1)
         d11 = sin(d10)
         d13bar = (two/(hy*hy*sqn))
         d8bar = d13bar*d11
         d9bar = cos(d10)*(d13bar*d8)*hy
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hy
         do 780 gi = 1, gp
            gfkny1(gi) = d13bar*sx(gi,1,j) + d2bar*sx(gi,1,j-1) +
     +                   d3bar*svpoty(gi,1,j-1) + d8bar*sy(gi,1,j-1) +
     +                   d9bar*svpoty(gi,1,j-1)
  780    continue
         fkiny1 = (two/(hy*hy*sqn))*(x(1,j)-d2*d5+d8*d11)
         d2 = y(1,j-1)
         d4 = hy*vpoty(1,j-1)
         d5 = cos(d4)
         d8 = x(1,j-1)
         d10 = hy*vpoty(1,j-1)
         d11 = sin(d10)
         d13bar = (two/(hy*hy*sqn))
         d8bar = -d13bar*d11
         d9bar = cos(d10)*(-d13bar*d8)*hy
         d2bar = -d13bar*d5
         d3bar = (-sin(d4)*(-d13bar*d2))*hy
         do 790 gi = 1, gp
            gfkny2(gi) = d13bar*sy(gi,1,j) + d2bar*sy(gi,1,j-1) +
     +                   d3bar*svpoty(gi,1,j-1) + d8bar*sx(gi,1,j-1) +
     +                   d9bar*svpoty(gi,1,j-1)
  790    continue
         fkiny2 = (two/(hy*hy*sqn))*(y(1,j)-d2*d5-d8*d11)
         sfac = sin(two*pi*vornum*(j-one)/dble(ny))
         cfac = cos(two*pi*vornum*(j-one)/dble(ny))
         do 800 gi = 1, gp
            yx(gi,1,j) = cfac*gfknx1(gi) + sfac*gfknx2(gi) +
     +                   gfkny1(gi) + yx(gi,1,j)
  800    continue
         do 810 gi = 1, gp
            yy(gi,1,j) = -sfac*gfknx1(gi) + cfac*gfknx2(gi) +
     +                   gfkny2(gi) + yy(gi,1,j)
  810    continue
         d7bar = (one/hx)
         d3bar = (one/hy)
         do 820 gi = 1, gp
            gffld(gi) = d3bar*svpotx(gi,1,j-1) +
     +                  (-d3bar*svpotx(gi,1,j)) +
     +                  d7bar*svpoty(gi,2,j-1) +
     +                  (-d7bar*svpoty(gi,1,j-1))
  820    continue
         ffield = (vpotx(1,j-1)-vpotx(1,j))/hy +
     +            (vpoty(2,j-1)-vpoty(1,j-1))/hx
         ffbar = (two*(tkappa**2)/sqn)
         do 830 gi = 1, gp
            gffld(gi) = ffbar*gffld(gi)
  830    continue
         ffield = (two*(tkappa**2)/sqn)*ffield
         ffbar = -(one/hy)
         do 840 gi = 1, gp
            yvpotx(gi,1,j) = ffbar*gffld(gi) + yvpotx(gi,1,j)
  840    continue
         d7bar = (one/hx)
         d3bar = (one/hy)
         do 850 gi = 1, gp
            gffld(gi) = d3bar*svpotx(gi,nx,j) +
     +                  (-d3bar*svpotx(gi,nx,j+1)) +
     +                  d7bar*svpoty(gi,nx+1,j) +
     +                  (-d7bar*svpoty(gi,nx,j))
  850    continue
         ffield = (vpotx(nx,j)-vpotx(nx,j+1))/hy +
     +            (vpoty(nx+1,j)-vpoty(nx,j))/hx
         ffbar = (two*(tkappa**2)/sqn)
         do 860 gi = 1, gp
            gffld(gi) = ffbar*gffld(gi)
  860    continue
         ffield = (two*(tkappa**2)/sqn)*ffield
         ffbar = (one/hx)
         do 870 gi = 1, gp
            yvpoty(gi,1,j) = ffbar*gffld(gi) + yvpoty(gi,1,j)
  870    continue
  880 continue

c     Kinetic energy part, at origin (only needed in zero field).

      d2 = x(1,1)
      d4 = hx*vpotx(1,1)
      d5 = cos(d4)
      d8 = y(1,1)
      d10 = hx*vpotx(1,1)
      d11 = sin(d10)
      d13bar = (two/(hx*hx*sqn))
      d8bar = d13bar*d11
      d9bar = cos(d10)*(d13bar*d8)*hx
      d2bar = -d13bar*d5
      d3bar = (-sin(d4)*(-d13bar*d2))*hx
      do 890 gi = 1, gp
         gfknx1(gi) = d13bar*sx(gi,2,1) + d2bar*sx(gi,1,1) +
     +                d3bar*svpotx(gi,1,1) + d8bar*sy(gi,1,1) +
     +                d9bar*svpotx(gi,1,1)
  890 continue
      fkinx1 = (two/(hx*hx*sqn))*(x(2,1)-d2*d5+d8*d11)
      d2 = y(1,1)
      d4 = hx*vpotx(1,1)
      d5 = cos(d4)
      d8 = x(1,1)
      d10 = hx*vpotx(1,1)
      d11 = sin(d10)
      d13bar = (two/(hx*hx*sqn))
      d8bar = -d13bar*d11
      d9bar = cos(d10)*(-d13bar*d8)*hx
      d2bar = -d13bar*d5
      d3bar = (-sin(d4)*(-d13bar*d2))*hx
      do 900 gi = 1, gp
         gfknx2(gi) = d13bar*sy(gi,2,1) + d2bar*sy(gi,1,1) +
     +                d3bar*svpotx(gi,1,1) + d8bar*sx(gi,1,1) +
     +                d9bar*svpotx(gi,1,1)
  900 continue
      fkinx2 = (two/(hx*hx*sqn))*(y(2,1)-d2*d5-d8*d11)
      d2 = x(1,1)
      d4 = hy*vpoty(1,1)
      d5 = cos(d4)
      d8 = y(1,1)
      d10 = hy*vpoty(1,1)
      d11 = sin(d10)
      d13bar = (two/(hy*hy*sqn))
      d8bar = d13bar*d11
      d9bar = cos(d10)*(d13bar*d8)*hy
      d2bar = -d13bar*d5
      d3bar = (-sin(d4)*(-d13bar*d2))*hy
      do 910 gi = 1, gp
         gfkny1(gi) = d13bar*sx(gi,1,2) + d2bar*sx(gi,1,1) +
     +                d3bar*svpoty(gi,1,1) + d8bar*sy(gi,1,1) +
     +                d9bar*svpoty(gi,1,1)
  910 continue
      fkiny1 = (two/(hy*hy*sqn))*(x(1,2)-d2*d5+d8*d11)
      d2 = y(1,1)
      d4 = hy*vpoty(1,1)
      d5 = cos(d4)
      d8 = x(1,1)
      d10 = hy*vpoty(1,1)
      d11 = sin(d10)
      d13bar = (two/(hy*hy*sqn))
      d8bar = -d13bar*d11
      d9bar = cos(d10)*(-d13bar*d8)*hy
      d2bar = -d13bar*d5
      d3bar = (-sin(d4)*(-d13bar*d2))*hy
      do 920 gi = 1, gp
         gfkny2(gi) = d13bar*sy(gi,1,2) + d2bar*sy(gi,1,1) +
     +                d3bar*svpoty(gi,1,1) + d8bar*sx(gi,1,1) +
     +                d9bar*svpoty(gi,1,1)
  920 continue
      fkiny2 = (two/(hy*hy*sqn))*(y(1,2)-d2*d5-d8*d11)
      d7bar = (one/hx)
      d3bar = (one/hy)
      do 930 gi = 1, gp
         gffld(gi) = d3bar*svpotx(gi,1,1) + (-d3bar*svpotx(gi,1,2)) +
     +               d7bar*svpoty(gi,2,1) + (-d7bar*svpoty(gi,1,1))
  930 continue
      ffield = (vpotx(1,1)-vpotx(1,2))/hy + (vpoty(2,1)-vpoty(1,1))/hx
      ffbar = (two*(tkappa**2)/sqn)
      do 940 gi = 1, gp
         gffld(gi) = ffbar*gffld(gi)
  940 continue
      ffield = (two*(tkappa**2)/sqn)*ffield
      d3 = hx*vpotx(1,1)
      d5 = -cos(d3)
      d9 = hx*vpotx(1,1)
      d11 = -sin(d9)
      d15 = hy*vpoty(1,1)
      d17 = -cos(d15)
      d21 = hy*vpoty(1,1)
      d23 = -sin(d21)
      d20bar = cos(d21)*(-fkiny2)*hy
      d14bar = (-sin(d15)*(-fkiny1))*hy
      d8bar = cos(d9)*(-fkinx2)*hx
      d2bar = (-sin(d3)*(-fkinx1))*hx
      do 950 gi = 1, gp
         yx(gi,1,1) = d5*gfknx1(gi) + d11*gfknx2(gi) + d17*gfkny1(gi) +
     +                d23*gfkny2(gi) + yx(gi,1,1) +
     +                d2bar*svpotx(gi,1,1) + d8bar*svpotx(gi,1,1) +
     +                d14bar*svpoty(gi,1,1) + d20bar*svpoty(gi,1,1)
  950 continue
      d3 = hx*vpotx(1,1)
      d4 = sin(d3)
      d8 = hx*vpotx(1,1)
      d10 = -cos(d8)
      d14 = hy*vpoty(1,1)
      d15 = sin(d14)
      d19 = hy*vpoty(1,1)
      d21 = -cos(d19)
      d18bar = (-sin(d19)*(-fkiny2))*hy
      d13bar = cos(d14)*fkiny1*hy
      d7bar = (-sin(d8)*(-fkinx2))*hx
      d2bar = cos(d3)*fkinx1*hx
      do 960 gi = 1, gp
         yy(gi,1,1) = d4*gfknx1(gi) + d10*gfknx2(gi) + d15*gfkny1(gi) +
     +                d21*gfkny2(gi) + yy(gi,1,1) +
     +                d2bar*svpotx(gi,1,1) + d7bar*svpotx(gi,1,1) +
     +                d13bar*svpoty(gi,1,1) + d18bar*svpoty(gi,1,1)
  960 continue
      d3 = hx*x(1,1)
      d5 = hx*vpotx(1,1)
      d6 = sin(d5)
      d9 = hx*y(1,1)
      d11 = hx*vpotx(1,1)
      d12 = cos(d11)
      d14 = d3*d6 + d9*d12
      d18 = hx*y(1,1)
      d20 = hx*vpotx(1,1)
      d21 = sin(d20)
      d24 = hx*x(1,1)
      d26 = hx*vpotx(1,1)
      d27 = cos(d26)
      d29 = d18*d21 - d24*d27
      ffbar = (one/hy)
      d25bar = (-sin(d26)*(-fkinx2*d24))*hx
      d23bar = -fkinx2*d27*hx
      d19bar = cos(d20)*(fkinx2*d18)*hx
      d17bar = fkinx2*d21*hx
      d10bar = (-sin(d11)*(fkinx1*d9))*hx
      d8bar = fkinx1*d12*hx
      d4bar = cos(d5)*(fkinx1*d3)*hx
      d2bar = fkinx1*d6*hx
      do 970 gi = 1, gp
         yvpotx(gi,1,1) = d14*gfknx1(gi) + d29*gfknx2(gi) +
     +                    ffbar*gffld(gi) + yvpotx(gi,1,1) +
     +                    d2bar*sx(gi,1,1) + d4bar*svpotx(gi,1,1) +
     +                    d8bar*sy(gi,1,1) + d10bar*svpotx(gi,1,1) +
     +                    d17bar*sy(gi,1,1) + d19bar*svpotx(gi,1,1) +
     +                    d23bar*sx(gi,1,1) + d25bar*svpotx(gi,1,1)
  970 continue
      d3 = hy*x(1,1)
      d5 = hy*vpoty(1,1)
      d6 = sin(d5)
      d9 = hy*y(1,1)
      d11 = hy*vpoty(1,1)
      d12 = cos(d11)
      d14 = d3*d6 + d9*d12
      d18 = hy*y(1,1)
      d20 = hy*vpoty(1,1)
      d21 = sin(d20)
      d24 = hy*x(1,1)
      d26 = hy*vpoty(1,1)
      d27 = cos(d26)
      d29 = d18*d21 - d24*d27
      ffbar = -(one/hx)
      d25bar = (-sin(d26)*(-fkiny2*d24))*hy
      d23bar = -fkiny2*d27*hy
      d19bar = cos(d20)*(fkiny2*d18)*hy
      d17bar = fkiny2*d21*hy
      d10bar = (-sin(d11)*(fkiny1*d9))*hy
      d8bar = fkiny1*d12*hy
      d4bar = cos(d5)*(fkiny1*d3)*hy
      d2bar = fkiny1*d6*hy
      do 980 gi = 1, gp
         yvpoty(gi,1,1) = d14*gfkny1(gi) + d29*gfkny2(gi) +
     +                    ffbar*gffld(gi) + yvpoty(gi,1,1) +
     +                    d2bar*sx(gi,1,1) + d4bar*svpoty(gi,1,1) +
     +                    d8bar*sy(gi,1,1) + d10bar*svpoty(gi,1,1) +
     +                    d17bar*sy(gi,1,1) + d19bar*svpoty(gi,1,1) +
     +                    d23bar*sx(gi,1,1) + d25bar*svpoty(gi,1,1)
  980 continue
      d2 = x(nx,1)
      d4 = hx*vpotx(nx,1)
      d5 = cos(d4)
      d8 = y(nx,1)
      d10 = hx*vpotx(nx,1)
      d11 = sin(d10)
      d13bar = (two/(hx*hx*sqn))
      d8bar = d13bar*d11
      d9bar = cos(d10)*(d13bar*d8)*hx
      d2bar = -d13bar*d5
      d3bar = (-sin(d4)*(-d13bar*d2))*hx
      do 990 gi = 1, gp
         gfknx1(gi) = d13bar*sx(gi,nx+1,1) + d2bar*sx(gi,nx,1) +
     +                d3bar*svpotx(gi,nx,1) + d8bar*sy(gi,nx,1) +
     +                d9bar*svpotx(gi,nx,1)
  990 continue
      fkinx1 = (two/(hx*hx*sqn))*(x(nx+1,1)-d2*d5+d8*d11)
      d2 = y(nx,1)
      d4 = hx*vpotx(nx,1)
      d5 = cos(d4)
      d8 = x(nx,1)
      d10 = hx*vpotx(nx,1)
      d11 = sin(d10)
      d13bar = (two/(hx*hx*sqn))
      d8bar = -d13bar*d11
      d9bar = cos(d10)*(-d13bar*d8)*hx
      d2bar = -d13bar*d5
      d3bar = (-sin(d4)*(-d13bar*d2))*hx
      do 1000 gi = 1, gp
         gfknx2(gi) = d13bar*sy(gi,nx+1,1) + d2bar*sy(gi,nx,1) +
     +                d3bar*svpotx(gi,nx,1) + d8bar*sx(gi,nx,1) +
     +                d9bar*svpotx(gi,nx,1)
 1000 continue
      fkinx2 = (two/(hx*hx*sqn))*(y(nx+1,1)-d2*d5-d8*d11)
      d2 = x(1,ny)
      d4 = hy*vpoty(1,ny)
      d5 = cos(d4)
      d8 = y(1,ny)
      d10 = hy*vpoty(1,ny)
      d11 = sin(d10)
      d13bar = (two/(hy*hy*sqn))
      d8bar = d13bar*d11
      d9bar = cos(d10)*(d13bar*d8)*hy
      d2bar = -d13bar*d5
      d3bar = (-sin(d4)*(-d13bar*d2))*hy
      do 1010 gi = 1, gp
         gfkny1(gi) = d13bar*sx(gi,1,ny+1) + d2bar*sx(gi,1,ny) +
     +                d3bar*svpoty(gi,1,ny) + d8bar*sy(gi,1,ny) +
     +                d9bar*svpoty(gi,1,ny)
 1010 continue
      fkiny1 = (two/(hy*hy*sqn))*(x(1,ny+1)-d2*d5+d8*d11)
      d2 = y(1,ny)
      d4 = hy*vpoty(1,ny)
      d5 = cos(d4)
      d8 = x(1,ny)
      d10 = hy*vpoty(1,ny)
      d11 = sin(d10)
      d13bar = (two/(hy*hy*sqn))
      d8bar = -d13bar*d11
      d9bar = cos(d10)*(-d13bar*d8)*hy
      d2bar = -d13bar*d5
      d3bar = (-sin(d4)*(-d13bar*d2))*hy
      do 1020 gi = 1, gp
         gfkny2(gi) = d13bar*sy(gi,1,ny+1) + d2bar*sy(gi,1,ny) +
     +                d3bar*svpoty(gi,1,ny) + d8bar*sx(gi,1,ny) +
     +                d9bar*svpoty(gi,1,ny)
 1020 continue
      fkiny2 = (two/(hy*hy*sqn))*(y(1,ny+1)-d2*d5-d8*d11)
      do 1030 gi = 1, gp
         yx(gi,1,1) = gfknx1(gi) + gfkny1(gi) + yx(gi,1,1)
 1030 continue
      do 1040 gi = 1, gp
         yy(gi,1,1) = gfknx2(gi) + gfkny2(gi) + yy(gi,1,1)
 1040 continue
      d7bar = (one/hx)
      d3bar = (one/hy)
      do 1050 gi = 1, gp
         gffld(gi) = d3bar*svpotx(gi,1,ny) +
     +               (-d3bar*svpotx(gi,1,ny+1)) +
     +               d7bar*svpoty(gi,2,ny) + (-d7bar*svpoty(gi,1,ny))
 1050 continue
      ffield = (vpotx(1,ny)-vpotx(1,ny+1))/hy +
     +         (vpoty(2,ny)-vpoty(1,ny))/hx
      ffbar = (two*(tkappa**2)/sqn)
      do 1060 gi = 1, gp
         gffld(gi) = ffbar*gffld(gi)
 1060 continue
      ffield = (two*(tkappa**2)/sqn)*ffield
      ffbar = -(one/hy)
      do 1070 gi = 1, gp
         yvpotx(gi,1,1) = ffbar*gffld(gi) + yvpotx(gi,1,1)
 1070 continue
      d7bar = (one/hx)
      d3bar = (one/hy)
      do 1080 gi = 1, gp
         gffld(gi) = d3bar*svpotx(gi,nx,1) + (-d3bar*svpotx(gi,nx,2)) +
     +               d7bar*svpoty(gi,nx+1,1) + (-d7bar*svpoty(gi,nx,1))
 1080 continue
      ffield = (vpotx(nx,1)-vpotx(nx,2))/hy +
     +         (vpoty(nx+1,1)-vpoty(nx,1))/hx
      ffbar = (two*(tkappa**2)/sqn)
      do 1090 gi = 1, gp
         gffld(gi) = ffbar*gffld(gi)
 1090 continue
      ffield = (two*(tkappa**2)/sqn)*ffield
      ffbar = (one/hx)
      do 1100 gi = 1, gp
         yvpoty(gi,1,1) = ffbar*gffld(gi) + yvpoty(gi,1,1)
 1100 continue

      end

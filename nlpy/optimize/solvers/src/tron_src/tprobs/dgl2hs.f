      subroutine dgl2hs(nx,ny,x,s,y,wa1,wa2,vornum)
      integer nx, ny, vornum
      double precision x(4*nx*ny), s(4*nx*ny), y(4*nx*ny),
     +                 wa1(4*(nx+1)*(ny+1)), wa2(4*(nx+1)*(ny+1))
c     **********
c
c     Subroutine dgl2hs
c
c     This subroutine computes the product f''(x)*s = y, where f''(x)
c     is the Hessian matrix for the Ginzburg-Landau (2-dimensional)
c     problem evaluted at x.
c
c     The subroutine statement is
c
c       subroutine dgl2hs(nx,ny,x,s,y,task,wa1,wa2,vornum)
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
c       x is a double precision array of dimension 4*nx*ny.
c         On entry x specifies the vector x.
c         On exit x is unchanged.
c
c       s is a double precision array of dimension 4*nx*ny.
c         On entry s contains the vector s.
c         On exit s is unchanged.
c
c       y is a double precision array of dimension 4*nx*ny.
c         On entry y need not be specified.
c         On exit y contains f''(x)*s.
c
c       wa1 is a double precision work array of dimension
c         4*(nx+1)(ny+1).
c
c       wa2 is a double precision work array of dimension
c         4*(nx+1)(ny+1).
c
c       vornum is an integer variable.
c         On entry vornum specifies the number of vortices.
c         On exit vornum is unchanged.
c
c     Subprograms called
c
c       MINPACK-2  ...  dgl2co
c
c     MINPACK-2 Project. August 1999.
c     Argonne National Laboratory.
c     Brett M. Averick
c
c     **********
      double precision zero
      parameter (zero=0.0d0)

      integer ctr, i, itemp, j, k

      external dgl2co

      itemp = (nx+1)*(ny+1)

c     Pack work array.

      ctr = 1
      do 20 j = 1, ny
         do 10 i = 1, nx
            k = (j-1)*nx + i
            wa2(ctr) = s(k)
            wa1(ctr) = x(k)
            wa2(itemp+ctr) = s(nx*ny+k)
            wa1(itemp+ctr) = x(nx*ny+k)
            wa2(2*itemp+ctr) = s(2*nx*ny+k)
            wa1(2*itemp+ctr) = x(2*nx*ny+k)
            wa2(3*itemp+ctr) = s(3*nx*ny+k)
            wa1(3*itemp+ctr) = x(3*nx*ny+k)
            ctr = ctr + 1
   10    continue
         wa1(ctr) = zero
         wa2(ctr) = zero
         wa1(itemp+ctr) = zero
         wa2(itemp+ctr) = zero
         wa1(2*itemp+ctr) = zero
         wa2(2*itemp+ctr) = zero
         wa1(3*itemp+ctr) = zero
         wa2(3*itemp+ctr) = zero
         ctr = ctr + 1
   20 continue

      call dgl2co(1,nx,ny,wa1(1),wa2(1),1,wa1(itemp+1),wa2(itemp+1),1,
     +            wa1(2*itemp+1),wa2(2*itemp+1),1,wa1(3*itemp+1),
     +            wa2(3*itemp+1),1,y(1),1,y(nx*ny+1),1,y(2*nx*ny+1),1,
     +            y(3*nx*ny+1),1,vornum)

      end

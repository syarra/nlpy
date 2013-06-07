      subroutine dminhs(n,nx,ny,x,s,hs,prob,par,w)
      character*6 prob
      integer n, nx, ny
      double precision par
      double precision x(n), s(n), hs(n), w(2*n)
c     **********
c
c     Subroutine dminhs
c
c     This subroutine computes the Hessian-vector product for
c     the minimization problem from the MINPACK-2 test problem
c     collection specified by the character variable prob.
c
c     The subroutine statement is
c
c       subroutine dminhs(n,nx,ny,x,s,hs,prob,par,w)
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
c       x is a double precision array of dimension n.
c         On entry x specifies the vector x.
c         On exit x is unchanged.
c
c       s is a double precision array of dimension n.
c         On entry s specifies a vector s.
c         On exit s is unchanged.
c
c       hs is a double precision array of dimension n.
c         On entry hs need not be specified.
c         On exit hs contains the product H*s where H is the
c            Hessian matrix at x.
c
c       prob is a character*6 variable.
c         On entry prob specifies the problem.
c         On exit prob is set to 'ERROR' if prob is not an
c            acceptable problem name. Otherwise prob is unchanged.
c
c       par is a double precision variable.
c         On entry par specifies a problem-dependent parameter.
c         On exit par is unchanged.
c
c       w is a double precision work array of dimension 2*n.
c
c     Subprograms called
c
c       MINPACK-2 ... depths, dgl1hs, dgl2hs, dmsahs, dmsabc,
c                     dodchs, dpjbhs, dsschs
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Brett M. Averick and Jorge J. More'.
c
c     **********
      double precision zero, one, ten
      parameter (zero=0.0d0,one=1.0d0,ten=10.0d0)

      external depths, dgl1hs, dgl2hs, dmsahs, dmsabc, dodchs, 
     +         dpjbhs, dsschs

c     Select a problem.

      if (prob(1:4) .eq. 'DEPT') then
         call depths(nx,ny,s,hs)
      else if (prob(1:4) .eq. 'DGL1') then
         call dgl1hs(n,x,s,hs,par)
       else if (prob(1:4) .eq. 'DGL2') then
         call dgl2hs(nx,ny,x,s,hs,w(1),w(4*(nx+1)*(ny+1)+1),int(par))
      else if (prob(1:4) .eq. 'DMSA') then
         call dmsabc(nx,ny,w(1),w(nx+3),w(2*nx+5),w(2*nx+ny+7))
         call dmsahs(nx,ny,x,s,hs,w(1),w(nx+3),w(2*nx+5),w(2*nx+ny+7))
      else if (prob(1:4) .eq. 'DMSO') then
         do i = 0, nx+1
            w(i+1) = one - (2*dble(i)/(nx+1) - one)**2
            w(nx+3+i) = w(i+1)
         end do
         do i = 0, ny+1
            w(2*nx+5+i) = zero
            w(2*nx+ny+7+i) = zero
         end do
         call dmsahs(nx,ny,x,s,hs,w(1),w(nx+3),w(2*nx+5),w(2*nx+ny+7))
      else if (prob(1:4) .eq. 'DODC') then
         call dodchs(nx,ny,x,s,hs,par)
      else if (prob(1:4) .eq. 'DPJB') then
         call dpjbhs(nx,ny,s,hs,par,ten)
      else if (prob(1:4) .eq. 'DSSC') then
         call dsschs(nx,ny,x,s,hs,par)
      else
         prob = 'ERROR'
      end if

      end

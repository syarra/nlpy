      subroutine dminfg(n,nx,ny,x,f,g,task,prob,par,w)
      character*6 prob
      character*60 task
      integer n, nx, ny
      double precision f, par
      double precision x(n), g(n), w(2*n)
c     *********
c
c     Subroutine dminfg
c
c     This subroutine computes the function and gradient for
c     the minimization problem from the MINPACK-2 test problem 
c     collection specified by the character variable prob.
c
c     The subroutine statement is
c
c       subroutine dminfg(n,nx,ny,x,f,g,task,prob,par,w)
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
c       f is a double precision variable.
c         On entry f need not be specified.
c         On exit f is set to the function evaluated at x if
c            task = 'F' or 'FG'.
c
c       g is a double precision array of dimension n.
c         On entry g need not be specified.
c         On exit g contains the gradient evaluated at x if
c            task = 'G' or 'FG'.
c
c       task is a character*60 variable.
c         On entry task specifies the action of the subroutine:
c
c            task               action
c            ----               ------
c             'F'     Evaluate the function at x.
c             'G'     Evaluate the gradient vector at x.
c             'FG'    Evaluate the function and the gradient at x.
c             'XS'    Set x to the standard starting point xs.
c
c         On exit task is unchanged.
c
c       prob is a character*6 variable.
c         On entry prob specifies the problem.
c         On exit prob is set to 'ERROR' if prob is not an
c            acceptable problem name. Otherwise prob is unchanged.
c
c       par is a double precision variable.
c         On entry par specifies a probem-dependent parameter.
c         On exit par is unchanged.
c
c       w is a double precision array of dimension 2*n.
c
c     Subprograms called
c
c       MINPACK-2 ... deptfg, dgl1fg, dgl2fg, dmsafg, dmsabc,
c                     dljcfg, dodcfg, dpjbfg, dsscfg
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory.
c     Brett M. Averick and Jorge J. More'.
c
c     **********
      double precision zero, one, ten
      parameter (zero=0.0d0,one=1.0d0,ten=10.0d0)

      integer i

      external deptfg, dgl1fg, dgl2fg, dmsafg, dmsabc, dljcfg, 
     +         dodcfg, dpjbfg, dsscfg

c     Select a problem.

      if (prob(1:4) .eq. 'DEPT') then
         call deptfg(nx,ny,x,f,g,task,par)
      else if (prob(1:4) .eq. 'DGL1') then
         call dgl1fg(n,x,f,g,task,par)
      else if (prob(1:4) .eq. 'DGL2') then
         call dgl2fg(nx,ny,x,f,g,task,w,int(par))
      else if (prob(1:4) .eq. 'DMSA') then
         call dmsabc(nx,ny,w(1),w(nx+3),w(2*nx+5),w(2*nx+ny+7))
         call dmsafg(nx,ny,x,f,g,task,w(1),w(nx+3),w(2*nx+5),
     +               w(2*nx+ny+7))
      else if (prob(1:4) .eq. 'DMSO') then
         do i = 0, nx+1
            w(i+1) = one - (2*dble(i)/(nx+1) - one)**2
            w(nx+3+i) = w(i+1)
         end do
         do i = 0, ny+1
            w(2*nx+5+i) = zero
            w(2*nx+ny+7+i) = zero
         end do
         call dmsafg(nx,ny,x,f,g,task,w(1),w(nx+3),w(2*nx+5),
     +               w(2*nx+ny+7))
      else if (prob(1:4) .eq. 'DLJ2') then
         call dljcfg(n,x,f,g,task,2,n/2)
      else if (prob(1:4) .eq. 'DLJ3') then
         call dljcfg(n,x,f,g,task,3,n/3)
      else if (prob(1:4) .eq. 'DODC') then
         call dodcfg(nx,ny,x,f,g,task,par)
      else if (prob(1:4) .eq. 'DPJB') then
         call dpjbfg(nx,ny,x,f,g,task,par,ten)
      else if (prob(1:4) .eq. 'DSSC') then
         call dsscfg(nx,ny,x,f,g,task,par)
      else
         prob = 'ERROR'
      end if

      end

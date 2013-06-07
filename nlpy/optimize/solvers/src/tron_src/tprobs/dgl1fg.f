      subroutine dgl1fg(n,x,f,fgrad,task,t)
      character*60 task
      integer n
      double precision f, t
      double precision x(n), fgrad(n)
c     **********
c
c     Subroutine dgl1fg
c
c     This subroutine computes the function and gradient of the
c     Ginzburg-Landau (1-dimensional) problem.
c
c     The subroutine statement is
c
c       subroutine dgl1fg(n,x,f,fgrad,task,t)
c
c     where
c
c       n is an integer variable.
c         On entry n is the number of subintervals in the domain.
c         On exit n is unchanged.
c
c       x is a double precision array of dimension n.
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
c       fgrad is a double precision array of dimension n.
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
c       t is a double precision variable.
c         On entry t is a temperature in (3.73,7.32).
c         On exit t is unchanged.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory and University of Minnesota.
c     Brett M. Averick, Cheryl Hile, and Jorge J. More'.
c
c     **********
      double precision four, one, ten, three, two, sxteen, zero
      parameter (zero=0.0d0,one=1.0d0,two=2.0d0,three=3.0d0,four=4.0d0,
     +          ten=1.0d1,sxteen=1.6d1)

      integer i, n1, n2
      double precision alphan, alphas, betan, betas, c, dn, ds, ec, em,
     +                 f1, f2, f3, fac, gamma, h1, h2, hbar, hcn, hcs,
     +                 penn, pens, pi, tcn, tcs, temp

c     Initialization.

c     Set electron mass (grams), speed of light (cm/sec), and
c     electronic charge (esu).

      em = 9.11d-28
      c = 2.99d+10
      ec = 4.80d-10

c     Set length of a half-layer of lead and tin (10**3-angstroms),
c     d = ds + dn.

      ds = 1.0d0
      dn = 2.2d0

c     Set critical temperature for lead and tin (Kelvin).

      tcs = 7.32d0
      tcn = 3.73d0

c     Set critical magnetic field for lead and tin at zero
c     temperature (gauss).

      hcs = 803.0d0
      hcn = 309.0d0

c     Set penetration depth for lead and tin at zero temperature
c     (cm).

      pens = 3.7d-6
      penn = 3.4d-6

c     Compute pi.

      pi = four*atan(one)

c     Set initial values for temperature dependent constants alphas,
c     alphan (ergs), and betas, betan (ergs-cm**3).

      alphas = -two*((ec/c)**2/em)*(hcs**2)*(pens**2)
      alphan = -two*((ec/c)**2/em)*(hcn**2)*(penn**2)
      betas = sxteen*pi*(((ec/c)**2/em)**2)*(hcs**2)*(pens**4)
      betan = sxteen*pi*(((ec/c)**2/em)**2)*(hcn**2)*(penn**4)

      alphas = alphas*((one-(t/tcs)**2)/(one+(t/tcs)**2))
      alphan = alphan*((one-(t/tcn)**2)/(one+(t/tcn)**2))
      betas = betas/((one+(t/tcs)**2)**2)
      betan = betan/((one+(t/tcn)**2)**2)

c     Set Planck's constant (erg-sec).

      hbar = 1.05459d-27

c     Set temperature dependent constant gamma (erg-cm**2).

      gamma = hbar**2/(four*em)

c     Scale temperature dependent constants to the same units.
c     This makes the order parameter dimensionless.

      fac = 1.0d6
      alphas = alphas*(fac**3)
      alphan = alphan*(fac**3)
      betas = betas*(fac**6)
      betan = betan*(fac**6)
      gamma = gamma*(fac**5)

c     Compute the number of subintervals in (-d,-ds), in (-ds,ds),
c     and in (ds,d).

      n1 = n/4
      n2 = n - 2*n1
      h1 = dn/dble(n1)
      h2 = (two*ds)/dble(n2)

c     Compute the standard starting point if task = 'XS'.

      if (task(1:2) .eq. 'XS') then
         temp = sqrt((betas+betan)/(two*(abs(alphas)+abs(alphan))))
         do 10 i = 1, n
            x(i) = temp
   10    continue

         return

      end if

      if (task(1:1) .eq. 'F' .or. task(1:2) .eq. 'FG') then
         f1 = zero
         f2 = zero
         f3 = zero
      end if
      if (task(1:1) .eq. 'G' .or. task(1:2) .eq. 'FG') then
         do 20 i = 1, n
            fgrad(i) = zero
   20    continue
      end if

c     Evaluate the function over the intervals (-d, -ds), (-ds, ds),
c     and (ds, d) if task = 'F' or task = 'FG'.

      if (task(1:1) .eq. 'F' .or. task(1:2) .eq. 'FG') then
         do 30 i = 1, n1
            f1 = f1 + (alphan/three)*(x(i+1)**2+x(i+1)*x(i)+x(i)**2) +
     +           (betan/ten)*(x(i+1)**4+x(i+1)**3*x(i)+
     +           x(i+1)**2*x(i)**2+x(i+1)*x(i)**3+x(i)**4) +
     +           gamma*((x(i+1)-x(i))/h1)**2
   30    continue
         do 40 i = n1 + 1, n1 + n2
            f2 = f2 + (alphas/three)*(x(i+1)**2+x(i+1)*x(i)+x(i)**2) +
     +           (betas/ten)*(x(i+1)**4+x(i+1)**3*x(i)+
     +           x(i+1)**2*x(i)**2+x(i+1)*x(i)**3+x(i)**4) +
     +           gamma*((x(i+1)-x(i))/h2)**2
   40    continue
         do 50 i = n1 + n2 + 1, n - 1
            f3 = f3 + (alphan/three)*(x(i+1)**2+x(i+1)*x(i)+x(i)**2) +
     +           (betan/ten)*(x(i+1)**4+x(i+1)**3*x(i)+
     +           x(i+1)**2*x(i)**2+x(i+1)*x(i)**3+x(i)**4) +
     +           gamma*((x(i+1)-x(i))/h1)**2
   50    continue

c        Special case for the right subinterval where x(n+1) = x(1).

         f3 = f3 + (alphan/three)*(x(1)**2+x(1)*x(n)+x(n)**2) +
     +        (betan/ten)*(x(1)**4+x(1)**3*x(n)+x(1)**2*x(n)**2+
     +        x(1)*x(n)**3+x(n)**4) + gamma*((x(1)-x(n))/h1)**2
         f = h1*f1 + h2*f2 + h1*f3
      end if

c     Evaluate the gradient over the intervals (-d, -ds), (-ds, ds),
c     and (ds, d).

      if (task(1:1) .eq. 'G' .or. task(1:2) .eq. 'FG') then
         do 60 i = 1, n1
            fgrad(i) = fgrad(i) + h1*((alphan/three)*(x(i+1)+two*x(i))+
     +                 (betan/ten)*(x(i+1)**3+two*(x(i+1)**2)*x(i)+
     +                 three*x(i+1)*(x(i)**2)+four*x(i)**3)-
     +                 gamma*(two/h1)*((x(i+1)-x(i))/h1))
            fgrad(i+1) = fgrad(i+1) + h1*
     +                   ((alphan/three)*(two*x(i+1)+x(i))+
     +                   (betan/ten)*(four*x(i+1)**3+three*(x(i+
     +                   1)**2)*x(i)+two*x(i+1)*(x(i)**2)+x(i)**3)+
     +                   gamma*(two/h1)*((x(i+1)-x(i))/h1))
   60    continue
         do 70 i = n1 + 1, n1 + n2
            fgrad(i) = fgrad(i) + h2*((alphas/three)*(x(i+1)+two*x(i))+
     +                 (betas/ten)*(x(i+1)**3+two*(x(i+1)**2)*x(i)+
     +                 three*x(i+1)*(x(i)**2)+four*x(i)**3)-
     +                 gamma*(two/h2)*((x(i+1)-x(i))/h2))
            fgrad(i+1) = fgrad(i+1) + h2*
     +                   ((alphas/three)*(two*x(i+1)+x(i))+
     +                   (betas/ten)*(four*x(i+1)**3+three*(x(i+
     +                   1)**2)*x(i)+two*x(i+1)*(x(i)**2)+x(i)**3)+
     +                   gamma*(two/h2)*((x(i+1)-x(i))/h2))
   70    continue
         do 80 i = n1 + n2 + 1, n - 1
            fgrad(i) = fgrad(i) + h1*((alphan/three)*(x(i+1)+two*x(i))+
     +                 (betan/ten)*(x(i+1)**3+two*(x(i+1)**2)*x(i)+
     +                 three*x(i+1)*(x(i)**2)+four*x(i)**3)-
     +                 gamma*(two/h1)*((x(i+1)-x(i))/h1))
            fgrad(i+1) = fgrad(i+1) + h1*
     +                   ((alphan/three)*(two*x(i+1)+x(i))+
     +                   (betan/ten)*(four*x(i+1)**3+three*(x(i+
     +                   1)**2)*x(i)+two*x(i+1)*(x(i)**2)+x(i)**3)+
     +                   gamma*(two/h1)*((x(i+1)-x(i))/h1))
   80    continue

c        Special case for the right subinterval where x(n+1) = x(1).

         fgrad(n) = fgrad(n) + h1*((alphan/three)*(x(1)+two*x(n))+
     +              (betan/ten)*(x(1)**3+two*(x(1)**2)*x(n)+
     +              three*x(1)*(x(n)**2)+four*x(n)**3)-
     +              gamma*(two/h1)*((x(1)-x(n))/h1))
         fgrad(1) = fgrad(1) + h1*((alphan/three)*(two*x(1)+x(n))+
     +              (betan/ten)*(four*x(1)**3+three*(x(1)**2)*x(n)+
     +              two*x(1)*(x(n)**2)+x(n)**3)+
     +              gamma*(two/h1)*((x(1)-x(n))/h1))
      end if

      end

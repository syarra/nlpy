      subroutine dgl1hs(n,x,s,y,t)
      integer n
      double precision t
      double precision x(n), s(n), y(n)
c     **********
c
c     Subroutine dglhs
c
c     This subroutine computes the product f''(x)*s = y, where
c     f''(x) is the Hessian matrix for the Ginzburg-Landau
c     (1-dimensional) problem.
c
c     The subroutine statement is
c
c       subroutine dgl1hs(n,x,s,y,t)
c
c     where
c
c       n is an integer variable.
c         On entry n is the number of variables.
c         On exit n is unchanged.
c
c       x is a double precision array of dimension n.
c         On entry x specifies the vector x.
c         On exit x is unchanged.
c
c       y is a double precision array of dimension n.
c         On entry out need not be specified.
c         On exit y contains f''(x)*s.
c
c       s is a double precision array of dimension n.
c         On entry s contains the vector s.
c         On exit s is unchanged.
c
c       t is a double precision variable.
c         On entry t is a temperature in (3.73,7.32).
c         On exit t is unchanged.
c
c     MINPACK-2 Project. March 1999.
c     Argonne National Laboratory and University of Minnesota.
c     Brett M. Averick and Jorge J. More'.
c
c     **********
      double precision four, one, six, sxteen, ten, three, twelve, two,
     +                 zero
      parameter (zero=0.0d0,one=1.0d0,two=2.0d0,three=3.0d0,four=4.0d0,
     +          six=6.0d0,ten=10.0d0,twelve=12.0d0,sxteen=16.0d0)

      integer i, n1, n2
      double precision alphan, alphas, betan, betas, c, dn, ds, ec, em,
     +                 fac, gamma, h1, h2, hbar, hcn, hcs, penn, pens,
     +                 pi, tcn, tcs

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

      do 10 i = 1, n
         y(i) = zero
   10 continue

c     Evaluate f''(x)*s over the intervals (-d, -ds), (-ds, ds),
c     and (ds, d).

      do 20 i = 1, n1
         y(i) = y(i) + h1*(two*alphan/three+
     +          (betan/ten)*(two*x(i+1)**2+six*x(i+1)*x(i)+
     +          twelve*x(i)**2)+two*gamma/h1/h1)*s(i) +
     +          h1*(alphan/three+(betan/ten)*
     +          (three*x(i+1)**2+four*x(i+1)*x(i)+three*x(i)**2)-
     +          two*gamma/h1/h1)*s(i+1)
         y(i+1) = y(i+1) + h1*(alphan/three+
     +            (betan/ten)*(three*x(i+1)**2+four*x(i+1)*x(i)+
     +            three*x(i)**2)-two*gamma/h1/h1)*s(i) +
     +            h1*(two*alphan/three+(betan/ten)*
     +            (twelve*x(i+1)**2+six*x(i+1)*x(i)+two*x(i)**2)+
     +            two*gamma/h1/h1)*s(i+1)
   20 continue
      do 30 i = n1 + 1, n1 + n2
         y(i) = y(i) + h2*(two*alphas/three+
     +          (betas/ten)*(two*x(i+1)**2+six*x(i+1)*x(i)+
     +          twelve*x(i)**2)+two*gamma/h2/h2)*s(i) +
     +          h2*(alphas/three+(betas/ten)*
     +          (three*x(i+1)**2+four*x(i+1)*x(i)+three*x(i)**2)-
     +          two*gamma/h2/h2)*s(i+1)
         y(i+1) = y(i+1) + h2*(alphas/three+
     +            (betas/ten)*(three*x(i+1)**2+four*x(i+1)*x(i)+
     +            three*x(i)**2)-two*gamma/h2/h2)*s(i) +
     +            h2*(two*alphas/three+(betas/ten)*
     +            (twelve*x(i+1)**2+six*x(i+1)*x(i)+two*x(i)**2)+
     +            two*gamma/h2/h2)*s(i+1)
   30 continue
      do 40 i = n1 + n2 + 1, n - 1
         y(i) = y(i) + h1*(two*alphan/three+
     +          (betan/ten)*(two*x(i+1)**2+six*x(i+1)*x(i)+
     +          twelve*x(i)**2)+two*gamma/h1/h1)*s(i) +
     +          h1*(alphan/three+(betan/ten)*
     +          (three*x(i+1)**2+four*x(i+1)*x(i)+three*x(i)**2)-
     +          two*gamma/h1/h1)*s(i+1)
         y(i+1) = y(i+1) + h1*(alphan/three+
     +            (betan/ten)*(three*x(i+1)**2+four*x(i+1)*x(i)+
     +            three*x(i)**2)-two*gamma/h1/h1)*s(i) +
     +            h1*(two*alphan/three+(betan/ten)*
     +            (twelve*x(i+1)**2+six*x(i+1)*x(i)+two*x(i)**2)+
     +            two*gamma/h1/h1)*s(i+1)
   40 continue

c     Special case for the right subinterval where x(n+1) = x(1).

      y(1) = y(1) + h1*(alphan/three+(betan/ten)*
     +       (three*x(1)**2+four*x(1)*x(n)+three*x(n)**2)-
     +       two*gamma/h1/h1)*s(n) + h1*(two*alphan/three+
     +       (betan/ten)*(twelve*x(1)**2+six*x(1)*x(n)+two*x(n)**2)+
     +       two*gamma/h1/h1)*s(1)
      y(n) = y(n) + h1*(two*alphan/three+
     +       (betan/ten)*(two*x(1)**2+six*x(1)*x(n)+twelve*x(n)**2)+
     +       two*gamma/h1/h1)*s(n) + h1*(alphan/three+
     +       (betan/ten)*(three*x(1)**2+four*x(1)*x(n)+three*x(n)**2)-
     +       two*gamma/h1/h1)*s(1)

      end

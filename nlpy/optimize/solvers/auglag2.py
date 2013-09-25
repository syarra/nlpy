"""
auglag2.py

Abstract classes of an augmented Lagrangian merit function and solver.
The class is compatable with both the standard and matrix-free NLP
definitions.
"""

import numpy as np
import logging
from nlpy.model.nlp import NLPModel
from nlpy.model.mfnlp import MFSlackNLP
from nlpy.optimize.solvers.lbfgs import LBFGS, LBFGS_structured
from nlpy.optimize.solvers.lsr1 import LSR1, LSR1_unrolling, LSR1_structured
from nlpy.optimize.solvers.lsqr import LSQRFramework
from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.krylov.linop import ReducedLinearOperator
from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
from nlpy.optimize.tr.trustregion import TrustRegionBQP as TRSolver
from nlpy.tools.exceptions import UserExitRequest
from nlpy.tools.utils import where
from nlpy.tools.timing import cputime


class AugmentedLagrangian(NLPModel):
    """
    This class is a reformulation of an NLP, used to compute the
    augmented Lagrangian function, gradient, and approximate Hessian in a
    method-of-multipliers optimization routine. Slack variables are introduced
    for inequality constraints and a function that computes the gradient
    projected on to variable bounds is included.
    """

    def __init__(self, nlp, **kwargs):

        # Analyze NLP to add slack variables to the formulation
        if not isinstance(nlp, MFSlackNLP):
            self.nlp = MFSlackNLP(nlp, keep_variable_bounds=True, **kwargs)
        else: self.nlp = nlp

        self.rho_init = kwargs.get('rho_init',10.)
        self.rho = self.rho_init

        self.pi0 = np.zeros(self.nlp.m)
        self.pi = self.pi0.copy()

        self.n = self.nlp.n
        self.m = 0
        self.Lvar = self.nlp.Lvar
        self.Uvar = self.nlp.Uvar
        self.x0 = self.nlp.x0

    def obj(self, x, **kwargs):
        """
        Evaluate augmented Lagrangian function.
        """
        cons = self.nlp.cons(x)

        alfunc = self.nlp.obj(x)
        alfunc -= np.dot(self.pi,cons)
        alfunc += 0.5*self.rho*np.dot(cons,cons)
        return alfunc

    def grad(self, x, **kwargs):
        """
        Evaluate augmented Lagrangian gradient.
        """
        nlp = self.nlp
        J = nlp.jac(x)
        cons = nlp.cons(x)
        algrad = nlp.grad(x) + J.T * ( -self.pi + self.rho * cons)
        return algrad

    def dual_feasibility(self, x, **kwargs):
        """
        Evaluate Lagrangian gradient.
        """
        nlp = self.nlp
        J = nlp.jac(x)
        lgrad = nlp.grad(x) - J.T * self.pi
        return lgrad


    def hprod(self, x, z, v, **kwargs):
        """
        Compute the Hessian-vector product of the Hessian of the augmented
        Lagrangian with arbitrary vector v.
        """
        nlp = self.nlp
        on = nlp.original_n
        om = nlp.original_m
        upperC = nlp.upperC ; nupperC = nlp.nupperC
        rangeC = nlp.rangeC ; nrangeC = nlp.nrangeC
        w = np.zeros(self.n)

        pi_bar = self.pi[:om].copy()
        pi_bar[upperC] *= -1.0
        pi_bar[rangeC] -= self.pi[om:].copy()

        cons = nlp.cons(x)
        mu = cons[:om].copy()
        mu[upperC] *= -1.0
        mu[rangeC] -= cons[om:].copy()

        w[:on] = nlp.hprod(x[:on],-pi_bar + self.rho * mu, v[:on])
        J = nlp.jac(x)
        w += self.rho * (J.T * (J * v))
        return w


    def hess(self, x, z=None, **kwargs):
        return SimpleLinearOperator(self.n, self.n, symmetric=True,
                                    matvec= lambda u: self.hprod(x,z,u))


class AugmentedLagrangianQuasiNewton(AugmentedLagrangian):
    """
    Use Quasi Newton approximate Hessian instead of true Hessian.
    """

    def __init__(self, nlp, **kwargs):
        AugmentedLagrangian.__init__(self, nlp, **kwargs)

    def hprod(self, x, z, v, **kwargs):
        """
        Compute the Hessian-vector product of the Hessian of the augmented
        Lagrangian with arbitrary vector v.
        """
        w = self.Hessapp.matvec(v)
        return w

    def hupdate(self, new_s=None, new_y=None):
        if new_s is not None and new_y is not None:
            self.Hessapp.store(new_s,new_y)
        return

    def hreset(self):
        self.Hessapp.restart()
        return


class AugmentedLagrangianLbfgs(AugmentedLagrangianQuasiNewton):
    """
    Use LBFGS approximate Hessian instead of true Hessian.
    """

    def __init__(self, nlp, **kwargs):
        AugmentedLagrangianQuasiNewton.__init__(self, nlp, **kwargs)
        self.Hessapp = LBFGS(self.n, npairs=kwargs.get('qn_pairs',5), scaling=True, **kwargs)



class AugmentedLagrangianLsr1(AugmentedLagrangianQuasiNewton):
    """
    Use an LSR1 approximation instead of the LBFGS approximation.
    """
    def __init__(self, nlp, **kwargs):
        AugmentedLagrangianQuasiNewton.__init__(self, nlp, **kwargs)
        self.Hessapp = LSR1(self.n, npairs=kwargs.get('qn_pairs',5), **kwargs)



class AugmentedLagrangianPartialQuasiNewton(AugmentedLagrangianQuasiNewton):
    """
    Only apply the Quasi Newton approximation to the second order terms of the 
    Hessian of the augmented Lagrangian, i.e. not the pJ'J term.
    """
    def __init__(self, nlp, **kwargs):
        AugmentedLagrangianQuasiNewton.__init__(self, nlp, **kwargs)


    def hprod(self, x, z, v, **kwargs):
        w = self.Hessapp.matvec(v)
        J = self.nlp.jac(x)
        w += self.rho * (J.T * (J * v))
        return w



class AugmentedLagrangianPartialLbfgs(AugmentedLagrangianPartialQuasiNewton):
    """
    Only apply the LBFGS approximation to the second order terms of the 
    Hessian of the augmented Lagrangian, i.e. not the pJ'J term.
    """
    def __init__(self, nlp, **kwargs):
        AugmentedLagrangianPartialQuasiNewton.__init__(self, nlp, **kwargs)
        self.Hessapp = LBFGS(self.n, npairs=kwargs.get('qn_pairs',1), scaling=True, **kwargs)



class AugmentedLagrangianPartialLsr1(AugmentedLagrangianPartialQuasiNewton):
    """
    Only apply the LSR1 approximation to the second order terms of the 
    Hessian of the augmented Lagrangian, i.e. not the pJ'J term.
    """
    def __init__(self, nlp, **kwargs):
        AugmentedLagrangianPartialQuasiNewton.__init__(self, nlp, **kwargs)
        self.Hessapp = LSR1_unrolling(self.n, npairs=kwargs.get('qn_pairs',min(3,self.n)), scaling=False, **kwargs)



class AugmentedLagrangianStructuredQuasiNewton(AugmentedLagrangianQuasiNewton):
    """
    Only apply the Quasi Newton approximation to the second order terms of the
    Hessian of the augmented Lagrangian, i.e. not the pJ'J term.

    H(x) = A(x) + \rho J(x)^T J(x)
    where A(x) is an approximation of the second order terms of the Hessian of
    the augmented Lagrangian.
    """
    def __init__(self, nlp, **kwargs):
        AugmentedLagrangian.__init__(self, nlp, **kwargs)

    def hprod(self, x, z, v, **kwargs):
        """
        Compute the Hessian-vector product of the Hessian of the augmented
        Lagrangian with arbitrary vector v.
        """
        w = self.Hessapp.matvec(v)
        J = self.nlp.jac(x)
        w += self.rho * (J.T * (J * v))
        return w

    def update(self, new_s=None, new_y=None, new_yd=None):
        if new_s is not None and new_y is not None and new_yd is not None:
            self.Hessapp.store(new_s,new_y,new_yd)
        return

    def hreset(self):
        self.Hessapp.restart()
        return



class AugmentedLagrangianStructuredLbfgs(AugmentedLagrangianStructuredQuasiNewton):
    """
    Only apply the LBFGS approximation to the second order terms of the 
    Hessian of the augmented Lagrangian, i.e. not the pJ'J term.
    """
    def __init__(self, nlp, **kwargs):
        AugmentedLagrangianStructuredQuasiNewton.__init__(self, nlp, **kwargs)
        self.Hessapp = LBFGS_structured(self.n,
                       npairs=kwargs.get('qn_pairs',100), scaling=True, **kwargs)



class AugmentedLagrangianStructuredLsr1(AugmentedLagrangianStructuredQuasiNewton):
    """
    Only apply the LSR1 approximation to the second order terms of the 
    Hessian of the augmented Lagrangian, i.e. not the pJ'J term.
    """
    def __init__(self, nlp, **kwargs):
        AugmentedLagrangianStructuredQuasiNewton.__init__(self, nlp, **kwargs)
        self.Hessapp = LSR1_structured(self.n,
                       npairs=kwargs.get('qn_pairs',100),#min(3,self.n)),
                       scaling=False, **kwargs)


class AugmentedLagrangianFramework(object):
    """
    Solve an NLP using the augmented Lagrangian method. This class is
    based on the successful Fortran code LANCELOT, but provides a more
    flexible implementation, along with some new features.


    `AUGLAG = AugmentedLagrangianFramework(nlp, InnerSolver)`

    :parameters:

        :nlp:   a :class:`NLPModel` object representing the problem. For
                instance, nlp may arise from an AMPL model
        :InnerSolver:   a  solver for solving the inner iteration subproblem

    :keywords:

        :x0:           starting point                    (default nlp.x0)
        :reltol:       relative stopping tolerance       (default `nlp.stop_d`)
        :abstol:       absolute stopping tolerance       (default 1.0e-6)
        :maxiter:      maximum number of iterations      (default max(1000,10n))
        :ny:           apply Nocedal/Yuan linesearch     (default False)
        :nbk:          max number of backtracking steps in Nocedal/Yuan
                       linesearch                        (default 5)
        :monotone:     use monotone descent strategy     (default False)
        :nIterNonMono: number of iterations for which non-strict descent can
                       be tolerated if monotone=False    (default 25)
        :logger_name:  name of a logger object that can be used in the post
                       iteration                         (default None)
        :verbose:      print log if True                 (default True)

    Once a `AugmentedLagrangianFramework` object has been instantiated and the
    problem is set up, solve problem by issuing a call to `AUGLAG.solve()`.
    The algorithm stops as soon as the infinity norm of the projected gradient
    of the Lagrangian falls below

        ``max(abstol, reltol * pg0)``

    where ``g0`` is the infinity norm of the projected gradient of the Lagrangian
    at the initial point.

    :Exit codes:
        :0:    Optimal solution found
        :-1:   Maximum iteration reached
        :-2:   Not making sufficient progress
        :-5:   Time limit exceeded

    References
    ----------
    [CGT91] A. R. Conn, N. I. M. Gould, and Ph. L. Toint, *LANCELOT: A Fortran
            Package for Large-Scale Nonlinear Optimization (Release A)*,
            Springer-Verlag, 1992

    [NoW06] J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd Edition
            Springer, 2006, pp 519--523
    """

    def __init__(self, nlp, innerSolver, **kwargs):
        """
        Initialize augmented Lagrangian method and options.
        Any options that are not used here are passed on into the bound-
        constrained solver.
        """

        self.alprob = AugmentedLagrangian(nlp,**kwargs)
        self.x = kwargs.get('x0', self.alprob.x0.copy())

        self.least_squares_pi = kwargs.get('least_squares_pi', False)

        self.innerSolver = innerSolver

        self.tau = kwargs.get('tau', 0.1)
        self.omega = None
        self.eta = None
        self.eta0 = 0.1258925
        self.omega0 = 1.
        self.omega_init = kwargs.get('omega_init',self.omega0*0.1) # rho_init**-1
        self.eta_init = kwargs.get('eta_init',self.eta0**0.1) # rho_init**-0.1
        self.a_omega = kwargs.get('a_omega',1.)
        self.b_omega = kwargs.get('b_omega',1.)
        self.a_eta = kwargs.get('a_eta',0.1)
        self.b_eta = kwargs.get('b_eta',0.9)
        self.omega_rel = kwargs.get('omega_rel',1.e-5)
        self.omega_abs = kwargs.get('omega_abs',1.e-7)
        self.eta_rel = kwargs.get('eta_rel',1.e-5)
        self.eta_abs = kwargs.get('eta_abs',1.e-7)

        self.f0 = self.f = None

        # Maximum number of total inner iterations 
        self.max_inner_iter = kwargs.get('max_inner_iter', 100*self.alprob.nlp.original_n)

        self.update_on_rejected_step = False

        self.inner_fail_count = 0
        self.status = None

        self.verbose = kwargs.get('verbose', True)
        self.hformat = '%-5s  %8s  %8s  %8s  %8s  %8s  %8s  %8s'
        self.header  = self.hformat % ('Iter','f(x)','|pg(x)|','eps_g',
                                       '|c(x)|','eps_c', 'penalty', 'sbmin')
        self.hlen   = len(self.header)
        self.hline  = '-' * self.hlen
        self.format = '%-5d  %8.1e  %8.1e  %8.1e  %8.1e  %8.1e  %8.1e  %5d'
        self.format0= '%-5d  %8.1e  %8.1e  %8s  %8s  %8s  %8s  %5s'

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlpy.auglag')
        self.log = logging.getLogger(logger_name)
        if not self.verbose:
            self.log.propagate=False


    def project(self, x, **kwargs):
        """
        Project the given x vector on to the bound-constrained space and 
        return the result. This function is useful when the starting point 
        of the optimization is not initially within the bounds.
        """
        return np.maximum(np.minimum(x,self.alprob.Uvar),self.alprob.Lvar)


    def project_gradient(self, x, g, **kwargs):
        """
        Project the provided gradient on to the bound-constrained space and
        return the result. This is a helper function for determining
        optimality conditions of the original NLP.
        """
        p = x - g
        med = np.maximum(np.minimum(p,self.alprob.Uvar),self.alprob.Lvar)
        q = x - med

        return q


    def magical_step(self, x, g, **kwargs):
        """
        Compute a "magical step" to improve the convergence rate of the 
        inner minimization algorithm. This step minimizes the augmented 
        Lagrangian with respect to the slack variables only for a fixed set 
        of decision variables.
        """
        alprob = self.alprob
        on = alprob.nlp.original_n
        m_step = np.zeros(alprob.n)
        m_step[on:] = -g[on:]/self.alprob.rho
        # Assuming slack variables are restricted to [0,+inf) interval
        m_step[on:] = np.where(-m_step[on:] > x[on:], -x[on:], m_step[on:])
        return m_step


    def get_active_bounds(self, x):
        """
        Returns a list containing the indices of variables that are at 
        either their lower or upper bound.
        """
        lower_active = where(x==self.alprob.Lvar)
        upper_active = where(x==self.alprob.Uvar)
        active_bound = np.concatenate((lower_active,upper_active))
        return active_bound


    def least_squares_multipliers(self, x, **kwargs):
        """
        Compute a least-squares estimate of the Lagrange multipliers for the 
        current point. This may lead to faster convergence of the augmented 
        Lagrangian algorithm, at the expense of more Jacobian-vector products.
        """
        nlp = self.alprob.nlp
        m = nlp.m
        n = nlp.n
        full_mult = kwargs.get('full_mult',False)

        lim = max(2*m,2*n)
        J = nlp.jac(x)

        # Determine which bounds are active to remove appropriate columns of J
        on_bound = self.get_active_bounds(x)
        not_on_bound = np.setdiff1d(np.arange(n, dtype=np.int), on_bound)
        Jred = ReducedLinearOperator(J, np.arange(m, dtype=np.int), 
            not_on_bound)

        if full_mult:
            g = nlp.grad(x)
        else:
            g = self.dual_feasibility(x)

        # Call LSQR method
        lsqr = LSQRFramework(Jred.T)
        lsqr.solve(g[not_on_bound], itnlim=lim)
        if lsqr.optimal:
            if full_mult:
                self.pi = lsqr.x.copy()
            else:
                self.pi += lsqr.x

        return


    def UpdateMultipliers(self, convals, status):
        """
        Infeasibility is sufficiently small; update multipliers and
        tighten feasibility and optimality tolerances
        """

        self.alprob.pi -= self.alprob.rho*convals
        if self.alprob.nlp.m != 0:
            self.log.debug('New multipliers = %g, %g' % (max(self.alprob.pi),min(self.alprob.pi)))

        if status == 'opt':
            # Safeguard: tighten tolerances only if desired optimality
            # is reached to prevent rapid decay of the tolerances from failed
            # inner loops
            self.eta /= self.alprob.rho**self.b_eta
            self.omega /= self.alprob.rho**self.b_omega
            self.inner_fail_count = 0
        else:
            self.inner_fail_count += 1
        return


    def UpdatePenaltyParameter(self):
        """
        Large infeasibility; increase rho and reset tolerances
        based on new rho.
        """
        self.alprob.rho /= self.tau
        self.eta = self.eta0*self.alprob.rho**-self.a_eta
        self.omega = self.omega0*self.alprob.rho**-self.a_omega
        return


    def PostIteration(self, **kwargs):
        """
        Override this method to perform additional work at the end of a 
        major iteration. For example, use this method to restart an 
        approximate Hessian.
        """
        return None


    def solve(self, **kwargs):
        """
        Solve the optimization problem and return the solution.
        """
        original_n = self.alprob.nlp.original_n

        # Move starting point into the feasible box
        self.x = self.project(self.x)

        # Use a least-squares estimate of the multipliers to start (if requested)
        if self.least_squares_pi and self.alprob.nlp.m != 0:
            self.least_squares_multipliers(self.x, full_mult=True)
            self.log.debug('New multipliers = %g, %g' % (max(self.alprob.pi),min(self.alprob.pi)))

        # First augmented lagrangian gradient evaluation
        dphi = self.alprob.grad(self.x)

        # "Smart" initialization of slack variables using the magical step 
        # function that is already available
        m_step_init = self.magical_step(self.x, dphi)
        self.x += m_step_init

        dL = self.alprob.dual_feasibility(self.x)
        self.f = self.f0 = self.alprob.nlp.obj(self.x[:original_n])

        PdL = self.project_gradient(self.x,dL)
        Pmax = np.max(np.abs(PdL))
        self.pg0 = self.pgnorm = Pmax

        # Specific handling for the case where the original NLP is
        # unconstrained
        if self.alprob.nlp.m == 0:
            max_cons = 0.
        else:
            max_cons = np.max(np.abs(self.alprob.nlp.cons(self.x)))
            cons_norm_ref = max_cons

        self.omega = self.omega_init
        self.eta = self.eta_init
        self.omega_opt = self.omega_rel * self.pg0 + self.omega_abs
        self.eta_opt = self.eta_rel * max_cons + self.eta_abs

        tr = TR(eta1=1.0e-4, eta2=.9, gamma1=.25, gamma2=2.5)

        self.iter = 0
        self.inner_fail_count = 0
        self.niter_total = 0
        infeas_iter = 0

        exitIter = False
        # Convergence check
        exitOptimal = ( Pmax <= self.omega_opt and max_cons <= self.eta_opt )

        t = cputime()

        # Print out header and initial log.
        self.log.info(self.hline)
        self.log.info(self.header)
        self.log.info(self.hline)
        self.log.info(self.format0 % (self.iter, self.f,
                                             self.pg0, '', max_cons,
                                             '', self.alprob.rho,''))

        while not (exitOptimal or exitIter):
            self.iter += 1

            # Perform bound-constrained minimization
            SBMIN = self.innerSolver(self.alprob, tr, TRSolver,
                                     reltol=self.omega, x0=self.x,
                                     #maxiter=self.max_inner_iter/10., verbose=True,
                                     update_on_rejected_step=self.update_on_rejected_step, **kwargs)

            SBMIN.Solve()
            self.x = SBMIN.x.copy()
            self.niter_total += SBMIN.iter

            dL = self.alprob.dual_feasibility(self.x)
            PdL = self.project_gradient(self.x,dL)
            Pmax_new = np.max(np.abs(PdL))
            convals_new = self.alprob.nlp.cons(self.x)

            # Specific handling for the case where the original NLP is
            # unconstrained
            if self.alprob.nlp.m == 0:
                max_cons_new = 0.
            else:
                max_cons_new = np.max(np.abs(convals_new))

            self.f = self.alprob.nlp.obj(self.x[:original_n])
            self.pgnorm = Pmax_new

            # Print out header, say, every 10 iterations (for readability)
            if self.iter % 1 == 10:
                self.log.info(self.hline)
                self.log.info(self.header)
                self.log.info(self.hline)

            self.log.info(self.format % (self.iter, self.f,
                          self.pgnorm, self.omega , max_cons_new,
                          self.eta, self.alprob.rho, SBMIN.iter))

            # Update penalty parameter or multipliers based on result
            if max_cons_new <= np.maximum(self.eta, self.eta_opt):

                # Update convergence check
                if max_cons_new <= self.eta_opt and Pmax_new <= self.omega_opt:
                    exitOptimal = True
                    break

                self.UpdateMultipliers(convals_new,SBMIN.status)

                # Update reference constraint norm on successful reduction
                cons_norm_ref = max_cons_new
                infeas_iter = 0

                # If optimality of the inner loop is not achieved within 10
                # major iterations, exit immediately
                if self.inner_fail_count == 10:
                    self.status = -3
                    self.log.debug('Current point could not be improved, exiting ... \n')
                    break

                self.log.debug('******  Updating multipliers estimates  ******\n')

            else:

                self.UpdatePenaltyParameter()
                self.log.debug('******  Keeping current multipliers estimates  ******\n')

                if max_cons_new > 0.99*cons_norm_ref and self.iter != 1:
                    infeas_iter += 1
                else:
                    cons_norm_ref = max_cons_new
                    infeas_iter = 0

                if infeas_iter == 10:
                    self.status = -2
                    self.log.debug('Problem appears to be infeasible, exiting ... \n')
                    break

            # Safeguard: tightest tolerance should be near optimality to prevent excessive
            # inner loop iterations at the end of the algorithm
            if self.omega < self.omega_opt:
                self.omega = self.omega_opt
            if self.eta < self.eta_opt:
                self.eta = self.eta_opt


            try:
                self.PostIteration()
            except UserExitRequest:
                self.status = -3

            exitIter = self.niter_total > self.max_inner_iter

        self.tsolve = cputime() - t    # Solve time
        if self.alprob.nlp.m != 0:
            self.pi_max = np.max(np.abs(self.alprob.pi))
            self.cons_max = np.max(np.abs(self.alprob.nlp.cons(self.x)))
        else:
            self.pi_max = None
            self.cons_max = None

        # Solution output, etc.
        if exitOptimal:
            self.status = 0
            self.log.debug('Optimal solution found \n')
        elif not exitOptimal and self.status is None:
            self.status = -1
            self.log.debug('Maximum number of iterations reached \n')

        self.log.info('f = %12.8g' % self.f)
        if self.alprob.nlp.m != 0:
            self.log.info('pi_max = %12.8g' % np.max(self.alprob.pi))
            self.log.info('max infeas. = %12.8g' % max_cons_new)




class AugmentedLagrangianQuasiNewtonFramework(AugmentedLagrangianFramework):

    def __init__(self,nlp, innerSolver, **kwargs):
        AugmentedLagrangianFramework.__init__(self, nlp, innerSolver, **kwargs)


    def PostIteration(self, **kwargs):
        """
        This method resets the Quasi Newton Hessian approximation.
        """
        self.alprob.hreset()
        return



class AugmentedLagrangianLbfgsFramework(AugmentedLagrangianQuasiNewtonFramework):

    def __init__(self,nlp, innerSolver, **kwargs):
        AugmentedLagrangianQuasiNewtonFramework.__init__(self, nlp, innerSolver, **kwargs)
        self.alprob = AugmentedLagrangianLbfgs(nlp,**kwargs)



class AugmentedLagrangianLsr1Framework(AugmentedLagrangianQuasiNewtonFramework):

    def __init__(self, nlp, innerSolver, **kwargs):
        AugmentedLagrangianQuasiNewtonFramework.__init__(self, nlp, innerSolver, **kwargs)
        self.alprob = AugmentedLagrangianLsr1(nlp,**kwargs)



class AugmentedLagrangianPartialLbfgsFramework(AugmentedLagrangianLbfgsFramework):

    def __init__(self, nlp, innerSolver, **kwargs):
        AugmentedLagrangianLbfgsFramework.__init__(self, nlp, innerSolver, **kwargs)
        self.alprob = AugmentedLagrangianPartialLbfgs(nlp,**kwargs)



class AugmentedLagrangianPartialLsr1Framework(AugmentedLagrangianQuasiNewtonFramework):

    def __init__(self, nlp, innerSolver, **kwargs):
        AugmentedLagrangianQuasiNewtonFramework.__init__(self, nlp, innerSolver, **kwargs)
        self.update_on_rejected_step = True
        self.alprob = AugmentedLagrangianPartialLsr1(nlp,**kwargs)



class AugmentedLagrangianStructuredLbfgsFramework(AugmentedLagrangianLbfgsFramework):

    def __init__(self, nlp, innerSolver, **kwargs):
        AugmentedLagrangianLbfgsFramework.__init__(self, nlp, innerSolver, **kwargs)
        self.alprob = AugmentedLagrangianStructuredLbfgs(nlp,**kwargs)



class AugmentedLagrangianStructuredLsr1Framework(AugmentedLagrangianQuasiNewtonFramework):

    def __init__(self, nlp, innerSolver, **kwargs):
        AugmentedLagrangianQuasiNewtonFramework.__init__(self, nlp, innerSolver, **kwargs)
        self.update_on_rejected_step = True
        self.alprob = AugmentedLagrangianStructuredLsr1(nlp,**kwargs)

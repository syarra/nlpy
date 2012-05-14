'''
auglag2.py

Abstract classes of an augmented Lagrangian merit function and solver.
The class is compatable with both the standard and matrix-free NLP
definitions.
'''

# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys
import pdb
import copy

# =============================================================================
# External Python modules
# =============================================================================
import numpy as np
import logging

# =============================================================================
# Extension modules
# =============================================================================
from nlpy.model.nlp import NLPModel
from nlpy.model.mfnlp import MFModel, SlackNLP
from nlpy.model.amplpy import AmplModel
from nlpy.optimize.solvers.lbfgs import LBFGS
from nlpy.optimize.solvers.lsr1 import LSR1
from nlpy.krylov.linop import PysparseLinearOperator, SimpleLinearOperator
from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
from nlpy.optimize.tr.trustregion import TrustRegionBQP as TRSolver
from nlpy.optimize.solvers.sbmin import SBMINFramework
from nlpy.optimize.solvers.sbmin import SBMINLbfgsFramework
from pysparse.sparse.pysparseMatrix import PysparseMatrix


class AugmentedLagrangian(NLPModel):
    '''
    This class is a reformulation of an NLP, used to compute the
    augmented Lagrangian function, gradient, and approximate Hessian in a
    method-of-multipliers optimization routine. Slack variables are introduced
    for inequality constraints and a function that computes the gradient
    projected on to variable bounds is included.
    '''

    def __init__(self, nlp, **kwargs):

        # Analyze NLP to add slack variables to the formulation
        if not isinstance(nlp, SlackNLP):
            self.nlp = SlackNLP(nlp, keep_variable_bounds=True, **kwargs)
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
    # end def


    def obj(self, x, **kwargs):
        '''
        Evaluate augmented Lagrangian function.
        '''
        cons = self.nlp.cons(x)

        alfunc = self.nlp.obj(x)

        alfunc -= np.dot(self.pi,cons)
        alfunc += 0.5*self.rho*np.dot(cons,cons)

        return alfunc
    # end def


    def grad(self, x, **kwargs):
        '''
        Evaluate augmented Lagrangian gradient.
        '''
        nlp = self.nlp
        J = nlp.jac(x)
        cons = nlp.cons(x)
        algrad = nlp.grad(x) + J.T * ( -self.pi + self.rho * cons)

        return algrad
    # end def


    def project_gradient(self, x, g, **kwargs):
        '''
        Project the provided gradient on to the bound-constrained space and
        return the result. This is a helper function for determining
        optimality conditions of the original NLP.
        '''

        p = x - g
        med = np.maximum(np.minimum(p,self.Uvar),self.Lvar)
        q = x - med

        return q


    def hprod(self, x, z, v, **kwargs):
        '''
        Compute the Hessian-vector product of the Hessian of the augmented
        Lagrangian with arbitrary vector v.
        '''
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
        mu = cons.copy()#[:om].copy()
        mu[upperC] *= -1.0
        mu[rangeC] -= cons[om:].copy()

        w[:on] = nlp.hprod(x[:on],-pi_bar + self.rho * mu, v[:on])
        J = nlp.jac(x)
        w += self.rho * (J.T * (J * v))

        return w


    def hess(self, x, z=None, **kwargs):
        return SimpleLinearOperator(self.n, self.n, symmetric=True,
                                    matvec= lambda u: self.hprod(x,z,u))
# end class



class AugmentedLagrangianLbfgs(AugmentedLagrangian):
    '''
    Use LBFGS approximate Hessian instead of true Hessian.
    '''

    def __init__(self, nlp, **kwargs):
        AugmentedLagrangian.__init__(self, nlp, **kwargs)
        self.Hessapp = LBFGS(self.n, npairs=5, **kwargs)


    def hprod(self, x, z, v, **kwargs):
        '''
        Compute the Hessian-vector product of the Hessian of the augmented
        Lagrangian with arbitrary vector v.
        '''
        w = self.Hessapp.matvec(v)
        return w


    def hess(self, x, z=None, **kwargs):
        return SimpleLinearOperator(self.n, self.n, symmetric=True,
                                    matvec= lambda u: self.hprod(x,z,u))


    def hupdate(self, new_s=None, new_y=None):
        if new_s is not None and new_y is not None:
            self.Hessapp.store(new_s,new_y)
        return


    def hrestart(self):
        self.Hessapp.restart()
        return


    def Hprod(self, x, z, v, **kwargs):
        '''
        Compute the Hessian-vector product of the Hessian of the augmented
        Lagrangian with arbitrary vector v.
        '''
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

        w[:on] = nlp.hprod(x[:on],pi_bar - self.rho * mu, v[:on])

        J = nlp.jac(x)
        w += self.rho * (J.T * (J * v))

        return w


    def Hess(self, x, z=None, **kwargs):
        return SimpleLinearOperator(self.n, self.n, symmetric=True,
                                    matvec= lambda u: self.Hprod(x,z,u))

# end class



class AugmentedLagrangianFramework(object):
    '''
    Solve an NLP using the augmented Lagrangian method. This class is
    based on the successful Fortran code LANCELOT, but provides a more
    flexible implementation.

    References
    ----------
    [CGT91] A. R. Conn, N. I. M. Gould, and Ph. L. Toint, *LANCELOT: A Fortran
            Package for Large-Scale Nonlinear Optimization (Release A)*,
            Springer-Verlag, 1992

    [NoW06] J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd Edition
            Springer, 2006, pp 519--523
    '''

    def __init__(self, nlp, innerSolver, **kwargs):

        '''
        Initialize augmented Lagrangian method and options.
        Any options that are not used here are passed on into the bound-
        constrained solver.
        '''

        self.alprob = AugmentedLagrangian(nlp,**kwargs)
        self.x = kwargs.get('x0', self.alprob.x0.copy())

        self.innerSolver = innerSolver

        self.phi0 = None
        self.dphi0 = None
        self.dphi0norm = None

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
        self.omega_opt = kwargs.get('omega_opt',1.e-7)
        self.eta_opt = kwargs.get('eta_opt',1.e-7)
        
        self.f0 = None
        self.f = +np.infty

        # Maximum number of outer iterations (use 'maxiter' or 'maxinner' keyword for TR)
        self.maxouter = kwargs.get('maxouter', 10*self.alprob.nlp.original_n)

        self.inner_fail_count = 0
        self.status = None

        self.verbose = kwargs.get('verbose', True)
        self.hformat = '%-5s  %8s  %8s  %8s  %8s  %8s  %8s  %8s'
        self.header  = self.hformat % ('Iter','f(x)','|pg(x)|','eps_g','|c(x)|','eps_c', 'penalty', 'sbmin')
        self.hlen   = len(self.header)
        self.hline  = '-' * self.hlen
        self.format = '%-5d  %8.1e  %8.1e  %8.1e  %8.1e  %8.1e  %8.1e  %5d'
        self.format0= '%-5d  %8.1e  %8.1e  %8s  %8s  %8s  %8s  %5s'

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlpy.auglag')
        self.log = logging.getLogger(logger_name)
        if not self.verbose:
            self.log.propagate=False


    def UpdateMultipliers(self, convals, status):

        '''
        Infeasibility is sufficiently small; update multipliers and 
        tighten feasibility and optimality tolerances
        '''
        self.alprob.pi -= self.alprob.rho*convals
        if status == 'opt':
            # Safeguard: tighten tolerances only if desired optimality 
            # is reached to prevent rapid decay of the tolerances from failed 
            # inner loops
            self.eta /= self.alprob.rho**self.b_eta
            self.omega /= self.alprob.rho**self.b_omega
            self.inner_fail_count = 0
        else:
            self.inner_fail_count += 1
        # end if
        return


    def UpdatePenaltyParameter(self):

        '''
        Large infeasibility; increase rho and reset tolerances 
        based on new rho.
        '''
        self.alprob.rho /= self.tau
        self.eta = self.eta0*self.alprob.rho**-self.a_eta
        self.omega = self.omega0*self.alprob.rho**-self.a_omega
        return


    def solve(self, **kwargs):

        '''
        Solve the optimization problem and return the solution.
        '''

        original_n = self.alprob.nlp.original_n

        # First function and gradient evaluation
        phi = self.alprob.obj(self.x)
        dphi = self.alprob.grad(self.x)
        self.f0 = self.alprob.nlp.obj(self.x[:original_n])

        Pdphi = self.alprob.project_gradient(self.x,dphi)
        Pmax = np.max(np.abs(Pdphi))

        # In case the original problem doesn't have constraints,
        # perform an sbmin minimization with given tolerances
        if self.alprob.nlp.m == 0:
            max_cons = 0
            self.omega = self.omega_opt
            self.eta = self.eta_opt
        else:
            max_cons = np.max(np.abs(self.alprob.nlp.cons(self.x)))
            self.omega = self.omega_init
            self.eta = self.eta_init

        self.iter = 0
        self.inner_fail_count = 0
        self.pg0 = Pmax
        self.niter_total = 0

        # Convergence check
        converged = Pmax <= self.omega_opt and max_cons <= self.eta_opt

        # Print out header and initial log.
        self.log.info(self.hline)
        self.log.info(self.header)
        self.log.info(self.hline)
        self.log.info(self.format0 % (self.iter, self.f,
                                             self.pg0, '', max_cons,
                                             '', self.alprob.rho,''))
        # While not converged, loop
        while not converged and self.iter < self.maxouter:

            self.iter += 1

            # Perform bound-constrained minimization
            tr = TR(eta1=1.0e-4, eta2=.99, gamma1=.3, gamma2=2.5)

            SBMIN = self.innerSolver(self.alprob, tr, TRSolver,
                                        reltol=self.omega, x0=self.x,
                                        verbose=True)

            SBMIN.Solve()
            self.x = SBMIN.x.copy()
            self.f = self.alprob.nlp.obj(self.x[:original_n])
            self.niter_total += SBMIN.iter

            dphi = self.alprob.grad(self.x)
            Pdphi = self.alprob.project_gradient(self.x,dphi)
            Pmax_new = np.max(np.abs(Pdphi))
            convals_new = self.alprob.nlp.cons(self.x)

            if self.alprob.nlp.m == 0:
                max_cons_new = 0
                self.iter = SBMIN.iter
            else:
                max_cons_new = np.max(np.abs(convals_new))

            self.f = self.alprob.nlp.obj(self.x[:original_n])
            self.pgnorm = Pmax_new

            # Print out header, say, every iteration (for readability)
            if self.iter % 1 == 0:
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
                    converged = True
                    break

                self.UpdateMultipliers(convals_new,SBMIN.status)

                # If optimality of the inner loop is not achieved within 10 
                # major iterations, exit immediately
                if self.inner_fail_count == 10:
                    self.status = 'Infeas'
                    self.log.debug('Current point could not be improved, exiting ... \n')
                    break

                self.log.debug('******  Updating multipliers estimates  ******\n')

            else:

                self.UpdatePenaltyParameter()
                self.log.debug('******  Keeping current multipliers estimates  ******\n')

            # end if

            # Safeguard: tightest tolerance should be near optimality to prevent excessive
            # inner loop iterations at the end of the algorithm
            if self.omega < self.omega_opt:
                self.omega = self.omega_opt
            if self.eta < self.eta_opt:
                self.eta = self.eta_opt

        # end while

        # Solution output, etc.
        if converged:
            self.status = 'Opt'
            self.log.debug('Optimal solution found \n')
        elif not converged and self.status is None:
            self.status = 'Iter'
            self.log.debug('Maximum number of iterations reached \n')

        self.log.info('f = %12.8g' % self.f)
        if self.alprob.nlp.m != 0:
            self.log.info('pi_max = %12.8g' % np.max(self.alprob.pi))
            self.log.info('max infeas. = %12.8g' % max_cons_new)

    # end def

# end class



class AugmentedLagrangianLbfgsFramework(AugmentedLagrangianFramework):

    def __init__(self,nlp, innerSolver, **kwargs):
        AugmentedLagrangianFramework.__init__(self, nlp, innerSolver, **kwargs)
        self.alprob = AugmentedLagrangianLbfgs(nlp)

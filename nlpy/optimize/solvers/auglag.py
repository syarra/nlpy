'''
auglag.py

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
import numpy

# =============================================================================
# Extension modules
# =============================================================================
from nlpy.model.nlp import NLPModel
from nlpy.model.mfnlp import MFModel
from nlpy.model.amplpy import AmplModel
from nlpy.optimize.solvers.lbfgs import LBFGS
from nlpy.optimize.solvers.lsr1 import LSR1
from nlpy.krylov.linop import PysparseLinearOperator
from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
from nlpy.optimize.tr.trustregion import TrustRegionBQP as TRSolver
from nlpy.optimize.solvers.sbmin import SBMINFramework
from nlpy.optimize.solvers.sbmin import SBMINLbfgsFramework
from pysparse.sparse.pysparseMatrix import PysparseMatrix

# =============================================================================
# Augmented Lagrangian Problem Class
# =============================================================================
class AugmentedLagrangian(NLPModel):
    '''
    This class is a reformulation of an NLP, used to compute the
    augmented Lagrangian function, gradient, and approximate Hessian in a
    method-of-multipliers optimization routine. Slack variables are introduced
    for inequality constraints and a function that computes the gradient
    projected on to variable bounds is included.

    Matrix-free NLP models are accomodated with the help of a Hessian
    approximation which can be updated and restarted via calls to methods in
    this class.
    '''

    def __init__(self, nlp, **kwargs):

        self.nlp = nlp

        # Temporary error message as the class does not yet support
        # range constraints
        # if nlp.nrangeC > 0:
        #     msg = 'Range inequality constraints are not supported.'
        #     raise ValueError, msg

        # Analyze NLP to add slack variables to the formulation
        # Ordering of the slacks in 'x' is assumed to be the order shown here
        self.nx = nlp.n
        self.nsLL = nlp.nlowerC
        self.nsUU = nlp.nupperC
        self.nsLR = nlp.nrangeC
        self.nsUR = nlp.nrangeC
        self.ns = nlp.nlowerC + nlp.nupperC + 2*nlp.nrangeC
        self.n = self.nx + self.ns
        self.m = 0

        # Copy initial data from NLP given new problem definition
        # Initialize slack variables to zero
        self.x0 = numpy.zeros(self.n,'d')
        self.x0[:self.nx] = nlp.x0

        # Need 2 multipliers for each range constraint identified
        self.pi0 = numpy.zeros(nlp.m + nlp.nrangeC,'d')
        self.pi0[:nlp.m] = nlp.pi0
        self.pi0[nlp.m:] = nlp.pi0[nlp.rangeC]
        self.pi = self.pi0

        self.rho_init = None
        self.rho = None

        # Create Hessian approximation by default
        self.approxHess = kwargs.get('approxHess',True)
        if self.approxHess:
            # LBFGS is currently the only option
            self.Hessapp = LBFGS(self.n,**kwargs)
            #self.Hessapp = LSR1(self.n,**kwargs)

        # Extend bound arrays to include slack variables
        self.Lvar = numpy.zeros(self.n,'d')
        self.Lvar[:self.nx] = nlp.Lvar

        self.Uvar = nlp.Infinity*numpy.ones(self.n,'d')
        self.Uvar[:self.nx] = nlp.Uvar

        # Bring in bound arrays for constraints and lists of constraint types
        self.Lcon = nlp.Lcon
        self.Ucon = nlp.Ucon
        self.lowerC = nlp.lowerC
        self.upperC = nlp.upperC
        self.equalC = nlp.equalC
        self.rangeC = nlp.rangeC

        # Needed for AMPL model
        #self.stop_d = nlp.stop_d

        # Saved values (private).
        self._last_x = numpy.infty * numpy.ones(self.n)
        self._last_obj = None
        self._last_infeas = None
        self._last_grad = None
        self._last_Hx = None


    # end def


    # Evaluate infeasibility measure (used in both objective and gradient)
    def get_infeas(self, x, **kwargs):
        if self._last_infeas is not None and (self._last_x == x).all():
            return self._last_infeas

        nx = self.nx
        nsLL_ind = nx + self.nsLL
        nsUU_ind = nsLL_ind + self.nsUU
        nsLR_ind = nsUU_ind + self.nsLR
        nsUR_ind = nsLR_ind + self.nsUR
        m = self.nlp.m

        # Infeasibility w.r.t. range lower bounds in the indices self.rangeC
        #                            upper bounds at the end of the array
        infeas = numpy.zeros(m + self.nlp.nrangeC)

        if m != 0:
            convals = self.nlp.cons(x[:nx])
            infeas[self.lowerC] = convals[self.lowerC] - x[nx:nsLL_ind] - self.Lcon[self.lowerC]
            infeas[self.upperC] = convals[self.upperC] + x[nsLL_ind:nsUU_ind] - self.Ucon[self.upperC]
            infeas[self.equalC] = convals[self.equalC] - self.Lcon[self.equalC]
            infeas[self.rangeC] = convals[self.rangeC] - x[nsUU_ind:nsLR_ind] - self.Lcon[self.rangeC]
            infeas[m:] = convals[self.rangeC] + x[nsLR_ind:nsUR_ind] - self.Ucon[self.rangeC]
        # end if

        if not (self._last_x == x).all():
            self._last_x = x.copy()
            self._last_obj = None   # Objective out of date
            self._last_grad = None  # Gradient out of date
            if not self.approxHess:
                self._last_Hx = None    # Hessian product out of date

        self._last_infeas = infeas
        return infeas
    # end def


    # Evaluate augmented Lagrangian function
    def obj(self, x, **kwargs):
        if self._last_obj is not None and (self._last_x == x).all():
            return self._last_obj

        nx = self.nx

        alfunc = self.nlp.obj(x[:nx])

        infeas = self.get_infeas(x)

        alfunc += numpy.dot(self.pi,infeas)
        alfunc += 0.5*self.rho*numpy.sum(infeas**2)

        self._last_obj = alfunc
        return alfunc
    # end def


    # Evaluate augmented Lagrangian gradient
    def grad(self, x, **kwargs):
        if self._last_grad is not None and (self._last_x == x).all():
            return self._last_grad

        nlp = self.nlp
        m = nlp.m
        nx = self.nx
        nsLL_ind = nx + self.nsLL
        nsUU_ind = nsLL_ind + self.nsUU
        nsLR_ind = nsUU_ind + self.nsLR
        nsUR_ind = nsLR_ind + self.nsUR

        algrad = numpy.zeros(self.n,'d')
        algrad[:nx] = nlp.grad(x[:nx])

        infeas = self.get_infeas(x)
        infeas_bar = infeas[:m].copy()
        infeas_bar[nlp.rangeC] += infeas[m:]
        pi_bar = self.pi[:m].copy()
        pi_bar[nlp.rangeC] += self.pi[m:]

        vec = pi_bar + self.rho*infeas_bar
        if isinstance(nlp, MFModel):
            algrad[:nx] += nlp.jtprod(x[:nx],vec)
        else:
            if isinstance(nlp, AmplModel):
                _JE = nlp.jac(x[:nx])
                JE = PysparseLinearOperator(_JE, symmetric=False)
                algrad[:nx] += JE.T * vec
            else:
                if m != 0:
                    J = nlp.jac(x[:nx])
                    algrad[:nx] += numpy.dot(J.T,vec)
        # end if

        algrad[nx:nsLL_ind] = -self.pi[nlp.lowerC] \
                              - self.rho*infeas[nlp.lowerC]
        algrad[nsLL_ind:nsUU_ind] = self.pi[nlp.upperC] \
                                    + self.rho*infeas[nlp.upperC]
        algrad[nsUU_ind:nsLR_ind] = -self.pi[nlp.rangeC] \
                                    - self.rho*infeas[nlp.rangeC]
        algrad[nsLR_ind:nsUR_ind] = self.pi[m:] + self.rho*infeas[m:]

        self._last_grad = algrad
        return algrad
    # end def


    def project_gradient(self, x, g, **kwargs):
        '''
        Project the provided gradient on to the bound-constrained space and
        return the result. This is a helper function for determining
        optimality conditions of the original NLP.
        '''

        p = x - g
        med = numpy.maximum(numpy.minimum(p,self.Uvar),self.Lvar)
        q = x - med

        return q


    def hprod(self, x, pi, v, **kwargs):
        '''
        Compute the Hessian-vector product of the Hessian of the augmented
        Lagrangian with arbitrary vector v. Both exact and approximate
        Hessians are supported.
        '''

        if self._last_Hx is not None and (self._last_x == x).all() and not self.approxHess:
            return self._last_Hx

        nlp = self.nlp
        m = nlp.m
        w = numpy.zeros(self.n,'d')
        nx = self.nx
        nsLL_ind = nx + self.nsLL
        nsUU_ind = nsLL_ind + self.nsUU
        nsLR_ind = nsUU_ind + self.nsLR
        nsUR_ind = nsLR_ind + self.nsUR

        if self.approxHess:
            # Approximate Hessian
            w = self.Hessapp.matvec(v)

        else:
            # Exact Hessian
            # Non-slack variables
            infeas = self.get_infeas(x)
            infeas_bar = infeas[:m]
            infeas_bar[nlp.rangeC] += infeas[m:]
            pi_bar = self.pi[:m]
            pi_bar[nlp.rangeC] += self.pi[m:]

            w[:nx] = nlp.hprod(x[:nx],pi_bar,v[:nx],**kwargs)

            for i in range(m):
                w[:nx] += (self.rho*infeas_bar[i]) \
                          * nlp.hiprod(i,x[:nx],v[:nx])
            # end for

            # Coupling between slacks and non-slacks
            if isinstance(nlp, MFModel):
                _v0 = self.rho*nlp.jprod(x[:nx],v[:nx])
                _v0[nlp.rangeC] *= 2.
                w[:nx] += nlp.jtprod(x[:nx],_v0)

                _v0[nlp.rangeC] /= 2.
                w[nx:nsLL_ind] -= _v0[nlp.lowerC]
                w[nsLL_ind:nsUU_ind] += _v0[nlp.upperC]
                w[nsUU_ind:nsLR_ind] -= _v0[nlp.rangeC]
                w[nsLR_ind:nsUR_ind] += _v0[nlp.rangeC]

                _v1 = numpy.zeros(m,'d')
                _v2 = numpy.zeros(m,'d')
                _v1[nlp.lowerC] = -self.rho*v[nx:nsLL_ind]
                _v1[nlp.upperC] = self.rho*v[nsLL_ind:nsUU_ind]
                _v1[nlp.rangeC] = -self.rho*v[nsUU_ind:nsLR_ind]
                _v2[nlp.rangeC] = self.rho*v[nsLR_ind:nsUR_ind]
                w[:nx] += nlp.jtprod(x[:nx],_v1) + nlp.jtprod(x[:nx],_v2)
            else:
                if isinstance(nlp, AmplModel):
                    _JE = nlp.jac(x[:nx])
                    JE = PysparseLinearOperator(_JE, symmetric=False)
                    w[:nx] += self.rho * (JE.T *(JE * v[:nx]))

                    _J = _JE[nlp.lowerC,:]
                    J = PysparseLinearOperator(_J, symmetric=False)
                    w[:nx] -= self.rho * (J.T * v[nx:nsLL_ind])
                    w[nx:nsLL_ind] -= self.rho * (J * v[:nx])

                    _J = _JE[nlp.upperC,:]
                    J = PysparseLinearOperator(_J, symmetric=False)
                    w[:nx] += self.rho * (J.T * v[nsLL_ind:nsUU_ind])
                    w[nsLL_ind:nsUU_ind] += self.rho * (J * v[:nx])

                    _J = _JE[nlp.rangeC,:]
                    J = PysparseLinearOperator(_J, symmetric=False)
                    w[:nx] -= self.rho * (J.T * v[nsUU_ind:nsLR_ind])
                    w[nsUU_ind:nsLR_ind] -= self.rho * (J * v[:nx])
                    w[:nx] += self.rho * (J.T * v[nsLR_ind:nsUR_ind])
                    w[nsLR_ind:nsUR_ind] += self.rho * (J * v[:nx])

                else:
                    J = nlp.jac(x[:nx])
                    w[:nx] += self.rho*numpy.dot(J.T,numpy.dot(J,v[:nx]))
                    w[:nx] -= self.rho*numpy.dot(J[nlp.lowerC,:].T,v[nx:nsLL_ind])
                    w[:nx] += self.rho*numpy.dot(J[nlp.upperC,:].T,v[nsLL_ind:nsUU_ind])
                    w[:nx] -= self.rho*numpy.dot(J[nlp.rangeC,:].T,v[nsUU_ind:nsLR_ind])
                    w[:nx] += self.rho*numpy.dot(J[nlp.rangeC,:].T,v[nsLR_ind:nsUR_ind])
                    w[nx:nsLL_ind] -= self.rho * \
                                      numpy.dot(J[nlp.lowerC,:],v[:nx])
                    w[nsLL_ind:nsUU_ind] += self.rho * \
                                            numpy.dot(J[nlp.upperC,:],v[:nx])
                    w[nsUU_ind:nsLR_ind] -= self.rho * \
                                      numpy.dot(J[nlp.rangeC,:],v[:nx])
                    w[nsLR_ind:nsUR_ind] += self.rho * \
                                            numpy.dot(J[nlp.rangeC,:],v[:nx])
                #end if
            # end if
            # Slack variables
            w[nx:] += self.rho*v[nx:]
            self._last_Hx = w
        # end if

        return w


    def hupdate(self, new_s=None, new_y=None):
        if self.approxHess and new_s is not None and new_y is not None:
            self.Hessapp.store(new_s,new_y)
        return

    def hrestart(self):
        if self.approxHess:
            self.Hessapp.restart()
        return

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

    def __init__(self, nlp, rho_init=10., tau=0.1, **kwargs):

        '''
        Initialize augmented Lagrangian method and options.
        Any options that are not used here are passed on into the bound-
        constrained solver.
        '''

        self.alprob = AugmentedLagrangian(nlp,**kwargs)
        self.x = kwargs.get('x0', self.alprob.x0.copy())
        self.pi0 = kwargs.get('pi0', numpy.zeros(self.alprob.nlp.m))
        self.phi0 = None
        self.dphi0 = None
        self.dphi0norm = None
        self.alprob.pi0 = self.pi0
        self.alprob.rho = numpy.array(rho_init)
        self.alprob.rho_init = numpy.array(rho_init) # Needed ?
        self.rho = self.alprob.rho
        self.pi = self.alprob.pi
        self.tau = tau
        self.omega = None
        self.eta = None
        self.omega_init = kwargs.get('omega_init',.1) # rho_init**-1
        self.eta_init = kwargs.get('eta_init',0.1) # rho_init**-0.1
        self.a_omega = kwargs.get('a_omega',1.)
        self.b_omega = kwargs.get('b_omega',1.)
        self.a_eta = kwargs.get('a_eta',0.1)
        self.b_eta = kwargs.get('b_eta',0.9)
        self.omega_opt = kwargs.get('omega_opt',1.e-9)
        self.eta_opt = kwargs.get('eta_opt',1.e-9)
        self.magic_steps = kwargs.get('magic_steps',True)

        self.f0 = None
        self.f = +numpy.infty

        # Maximum number of outer iterations (use maxiter or maxinner for TR)
        self.maxouter = kwargs.get('maxouter', 1000)
        self.printlevel = kwargs.get('printlevel', 1)

        if self.printlevel > 1:
            self.sbmin_verbose = True
        else: self.sbmin_verbose = False
        # (Future: create a logger to track problem data and print)

    # end def


    def solve(self, **kwargs):

        '''
        Solve the optimization problem and return the solution.

        Currently, only equality constraints are supported in the optimization
        problem formulation.
        '''

        # First function and gradient evaluation
        phi = self.alprob.obj(self.x)
        dphi = self.alprob.grad(self.x)
        self.f0 = self.alprob.nlp.obj(self.x[:self.alprob.nx])

        Pdphi = self.alprob.project_gradient(self.x,dphi)
        Pmax = numpy.max(numpy.abs(Pdphi))

        # In case the original problem doesn't have constraint
        # perform a sbmin minimization with given tolerances
        if self.alprob.nlp.m == 0:
            max_cons = 0
            self.omega = self.omega_opt
            self.eta = self.eta_opt
        else:
            max_cons = numpy.max(numpy.abs(self.alprob.get_infeas(self.x)))
            self.omega = self.omega_init
            self.eta = self.eta_init

        self.iter = 0
        self.inner_fail_count = 0
        self.pg0 = Pmax


        # Convergence check
        converged = Pmax <= self.omega_opt and max_cons <= self.eta_opt

        # While not converged, loop
        while not converged and self.iter < self.maxouter:

            self.iter += 1

            if self.printlevel>=1:
                print 'Major iteration : ', self.iter
                print 'Penalty parameter : %6.3e'%self.rho
                print 'Required Projected gradient norm = %6.3e'%self.omega
                print 'Required constraint         norm = %6.3e'%self.eta
                print  '\n'

            # Perform bound-constrained minimization
            tr = TR(eta1=0.25, eta2=0.75, gamma1=0.0625, gamma2=2)

            if self.alprob.approxHess==True:
                SBMIN = SBMINLbfgsFramework(self.alprob, tr, TRSolver,
                                            reltol=self.omega, x0=self.x,
                                            f0=phi, g0=dphi, maxiter = 1000,
                                            verbose=self.sbmin_verbose,
                                            magic_steps=self.magic_steps)
            else:
                SBMIN = SBMINFramework(self.alprob, tr, TRSolver,
                                        reltol=self.omega, x0=self.x,
                                        f0=phi, g0=dphi, maxiter = 1000,
                                        verbose=self.sbmin_verbose,
                                        magic_steps=self.magic_steps)

            SBMIN.Solve(rho_pen=self.rho,slack_index=self.alprob.nx)

            # Retrieve solution from SBMIN
            self.x = SBMIN.x
            phi = SBMIN.f
            dphi = SBMIN.g

            self.alprob.hrestart
            Pdphi = self.alprob.project_gradient(self.x,dphi)
            Pmax_new = numpy.max(numpy.abs(Pdphi))
            convals_new = self.alprob.get_infeas(self.x)

            if self.alprob.nlp.m == 0:
                max_cons_new = 0
            else:
                max_cons_new = numpy.max(numpy.abs(convals_new))

            self.f = self.alprob.nlp.obj(self.x[:self.alprob.nx])
            self.pgnorm = Pmax_new

            if self.printlevel>=1:
                print ''
                print 'Objective function value  =  %e'%self.f
                print 'Penalty parameter         =  %6.4e'%self.rho
                print 'Projected gradient norm   =  %6.4e'%Pmax_new, \
                      ' Required gradient   norm =  %6.4e'%self.omega
                print 'Constraint         norm   =  %6.4e'%max_cons_new, \
                      ' Required constraint norm =  %6.4e'%self.eta


            # Update penalty parameter or multipliers based on result
            if max_cons_new <= numpy.maximum(self.eta, self.eta_opt):

                # Update convergence check
                if max_cons_new <= self.eta_opt and Pmax_new <= self.omega_opt:
                    converged = True
                    break
                # No change in rho, update multipliers, tighten tolerances
                self.pi += self.rho*convals_new
                if SBMIN.status == 'opt':
                    # Safeguard: tighten tolerances only if desired optimality 
                    # is reached to prevent rapid decay of tolerances from failed
                    # inner loops
                    self.eta /= self.rho**self.b_eta
                    self.omega /= self.rho**self.b_omega
                    self.inner_fail_count = 0
                else:
                    self.inner_fail_count += 1
                    if self.inner_fail_count == 10:
                        print '\n Current point could not be improved, exiting ... \n'
                        break
                # end if
                if self.printlevel>=1:
                    print '\n******  Updating multiplier estimates  ******\n'
            else:
                # Increase rho, reset tolerances based on new rho
                self.rho /= self.tau
                self.eta = self.eta_init*self.rho**-self.a_eta
                self.omega = self.omega_init*self.rho**-self.a_omega
                if self.printlevel>=1:
                    print '\n******  Keeping current multiplier estimates  ******\n'
            # end if

            # Safeguard: tightest tolerance should be near optimality to prevent excessive
            # inner loop iterations at the end of the algorithm
            if self.omega < self.omega_opt:
                self.omega = self.omega_opt
            if self.eta < self.eta_opt:
                self.eta = self.eta_opt

            # Update function and gradient calls for next iteration
            phi = self.alprob.obj(self.x)
            dphi = self.alprob.grad(self.x)

        # end while

        # Solution output, etc.
        if converged:
            print '\n Optimal solution found \n'

        if self.printlevel>=1:
            print 'f = ',self.f
            if self.alprob.nlp.m != 0:
                print 'pi_max = ',numpy.max(self.pi)
                print 'max infeas. = ',max_cons_new

    # end def



# end class

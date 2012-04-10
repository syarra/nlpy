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
        if nlp.nrangeC > 0:
            msg = 'Range inequality constraints are not supported.'
            raise ValueError, msg

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

        self.pi0 = nlp.pi0.copy()
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

        # Needed for AMPL model
        #self.stop_d = nlp.stop_d

    # end def


    # Evaluate infeasibility measure (used in both objective and gradient)
    def get_infeas(self, x, **kwargs):
        nx = self.nx
        nsLL_ind = nx + self.nsLL
        nsUU_ind = nsLL_ind + self.nsUU
        nsLR_ind = nsUU_ind + self.nsLR
        nsUR_ind = nsLR_ind + self.nsUR

        if self.nlp.m == 0:
            convals = numpy.zeros(self.nlp.m)
        else:
            convals = self.nlp.cons(x[:nx])
            #convals[self.nlp.m:] = convals[self.nlp.rangeC]
            convals[self.lowerC] -= x[nx:nsLL_ind] + self.Lcon[self.lowerC]
            convals[self.upperC] += x[nsLL_ind:nsUU_ind] - self.Ucon[self.upperC]
            convals[self.equalC] -= self.Lcon[self.equalC]
        # end if

        return convals
    # end def


    # Evaluate augmented Lagrangian function
    def obj(self, x, **kwargs):
        nx = self.nx
        nsLL_ind = nx + self.nsLL
        nsUU_ind = nsLL_ind + self.nsUU
        # nsLR_ind = nsUU_ind + self.nsLR
        # nsUR_ind = nsLR_ind + self.nsUR

        alfunc = self.nlp.obj(x[:nx])

        convals = self.get_infeas(x)

        alfunc += numpy.dot(self.pi,convals)
        alfunc += 0.5*self.rho*numpy.sum(convals**2)

        return alfunc
    # end def


    # Evaluate augmented Lagrangian gradient
    def grad(self, x, **kwargs):
        nlp = self.nlp
        nx = self.nx
        nsLL_ind = nx + self.nsLL
        nsUU_ind = nsLL_ind + self.nsUU
        # nsLR_ind = nsUU_ind + self.nsLR
        # nsUR_ind = nsLR_ind + self.nsUR

        algrad = numpy.zeros(self.n,'d')
        algrad[:nx] = nlp.grad(x[:nx])

        convals = self.get_infeas(x)

        vec = self.pi + self.rho*convals
        if isinstance(nlp, MFModel):
            algrad[:nx] += nlp.jtprod(x[:nx],vec)
        else:
            if isinstance(nlp, AmplModel):
                _JE = nlp.jac(x[:nx])
                JE = PysparseLinearOperator(_JE, symmetric=False)
                algrad[:nx] += JE.T * vec
            else:
                if self.nlp.m != 0:
                    algrad[:nx] += numpy.dot(nlp.jac(x[:nx]).transpose(),vec)
        # end if

        algrad[nx:nsLL_ind] = -self.pi[nlp.lowerC] \
                              - self.rho*convals[nlp.lowerC]
        algrad[nsLL_ind:nsUU_ind] = self.pi[nlp.upperC] \
                                    + self.rho*convals[nlp.upperC]
        # **Range constraint slacks here**

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

        nlp = self.nlp
        w = numpy.zeros(self.n,'d')
        nx = self.nx
        nsLL_ind = nx + self.nsLL
        nsUU_ind = nsLL_ind + self.nsUU
        # nsLR_ind = nsUU_ind + self.nsLR
        # nsUR_ind = nsLR_ind + self.nsUR


        # Non-slack variables
        if self.approxHess:
            # Approximate Hessian
            w = self.Hessapp.matvec(v)

        else:
            # Exact Hessian
            # Note: the code in this block has yet to be properly tested
            convals = self.get_infeas(x)
            w[:nx] = nlp.hprod(x[:nx],self.pi,v[:nx],**kwargs)

            for i in range(nlp.m):
                w[:nx] += (self.pi[i] + self.rho*convals[i]) \
                          * nlp.hiprod(x[:nx],i,v[:nx])
            # end for
            if isinstance(nlp, MFModel):
                w[:nx] += self.rho*nlp.jtprod(x[:nx],nlp.jprod(x[:nx],v[:nx]))
                w[:nx] += self.rho*nlp.jprod(x[:nx],v[:nx])
                w[nx:] += self.rho*nlp.jtprod(x[:nx],v[nx:])
            else:
                if isinstance(nlp, AmplModel):
                    _JE = nlp.jac(x[:nx])
                    JE = PysparseLinearOperator(_JE, symmetric=False)
                    w[:nx] += self.rho * (JE.T *(JE * v[:nx]))

                    _J = nlp.jac(x[:nx])[nlp.lowerC,:]
                    J = PysparseLinearOperator(_J, symmetric=False)
                    w[:nx] -= self.rho * (J.T * v[nx:nsLL_ind])
                    w[nx:nsLL_ind] -= self.rho * (J * v[:nx])

                    _J = nlp.jac(x[:nx])[nlp.lowerC,:]
                    J = PysparseLinearOperator(_J, symmetric=False)
                    w[:nx] += self.rho * (J.T * v[nsLL_ind:nsUU_ind])
                    w[nsLL_ind:nsUU_ind] += self.rho * (J * v[:nx])

                else:
                    J = nlp.jac(x[:nx])
                    w[:nx] += self.rho*numpy.dot(J.T,numpy.dot(J,v[:nx]))
                    w[:nx] -= self.rho*numpy.dot(J[nlp.lowerC].T,v[nx:nsLL_ind])
                    w[:nx] += self.rho*numpy.dot(J[nlp.upperC].T,v[nsLL_ind:nsUU_ind])
                    w[nx:nsLL_ind] -= self.rho * \
                                      numpy.dot(J[nlp.lowerC],v[:nx])
                    w[nsLL_ind:nsUU_ind] += self.rho * \
                                            numpy.dot(J[nlp.upperC],v[:nx])
                #end if
            # end if
            # Slack variables
            w[nx:nsLL_ind] -= self.rho*v[nx:nsLL_ind]
            w[nsLL_ind:nsUU_ind] += self.rho*v[nsLL_ind:nsUU_ind]
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
                                            maxiter = 1000,
                                            verbose=self.sbmin_verbose)
            else:
                SBMIN = SBMINFramework(self.alprob, tr, TRSolver,
                                       reltol=self.omega, x0=self.x,
                                       maxiter = 1000,
                                       verbose=self.sbmin_verbose)

            SBMIN.Solve()
            self.x = SBMIN.x
            self.f = self.alprob.nlp.obj(self.x[:self.alprob.nx])

            self.alprob.hrestart
            dphi = self.alprob.grad(self.x)
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
                # No change in rho, tighten tolerances
                self.pi += self.rho*convals_new
                if SBMIN.status == 'opt':
                    # Safeguard: tighten tolerances only if desired optimality 
                    # is reached to prevent rapid decay of tolerances
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
                    print '\n******  Updating multipliers estimates  ******\n'
            else:
                # Increase rho, reset tolerances based on new rho
                self.rho /= self.tau
                self.eta = self.eta_init*self.rho**-self.a_eta
                self.omega = self.omega_init*self.rho**-self.a_omega
                if self.printlevel>=1:
                    print '\n******  Keeping current multipliers estimates  ******\n'
            # end if

        # end while

        # Solution output, etc.
        if converged:
            print '\n Optimal solution found \n'

        if self.printlevel>=1:
            print 'f = ',self.f
            if self.alprob.m != 0:
                print 'pi_max = ',numpy.max(self.pi)
                print 'max infeas. = ',max_cons_new

    # end def



# end class

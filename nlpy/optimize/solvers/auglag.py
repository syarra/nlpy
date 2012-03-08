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
        if nlp.nrangeC:
            msg = 'Range inequality constraints are not supported.'
            raise ValueError, msg

        # Analyze NLP to add slack variables to the formulation
        # Ordering of the slacks in 'x' is assumed to be the order shown here
        self.nx = nlp.n
        self.nsLL = nlp.nlowerC
        self.nsUU = nlp.nupperC
        # self.nsLR = nlp.nrangeC
        # self.nsUR = nlp.nrangeC
        self.ns = nlp.nlowerC + nlp.nupperC + 2*nlp.nrangeC
        self.n = self.nx + self.ns
        self.m = nlp.m 

        # Copy initial data from NLP given new problem definition
        # Initialize slack variables to zero
        self.x0 = numpy.zeros(self.n,'d')
        self.x0[:self.nx] = nlp.x0

        self.pi0 = nlp.pi0.copy()

        # Create Hessian approximation by default
        self.approxHess = kwargs.get('approxHess',True)
        if self.approxHess:
        	# LBFGS is currently the only option
        	self.Hessapp = LBFGS(self.nx)

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
        # self.rangeC = nlp.rangeC
        self.equalC = nlp.equalC

	# end def 


	# Evaluate infeasibility measure (used in both objective and gradient)
	def get_infeas(self, x, **kwargs):
		convals = self.nlp.cons(x[:nx])
		convals[self.lowerC] -= x[nx:nsLL_ind] + self.Lcon[self.lowerC]
		convals[self.upperC] += x[nsLL_ind:nsUU_ind] - self.Ucon[self.upperC]
		convals[self.equalC] -= self.Lcon[self.equalC]
		# convals[self.rangeC] += x[nsLR_ind:nsUR_ind] - x[nsUU_ind:nsLR_ind]
		return convals
	# end def


	# Evaluate augmented Lagrangian function
	def obj(self, x, pi, rho, **kwargs):
		nx = self.nx
		nsLL_ind = nx + self.nsLL
		nsUU_ind = nsLL_ind + self.nsUU
		# nsLR_ind = nsUU_ind + self.nsLR
		# nsUR_ind = nsLR_ind + self.nsUR

		alfunc = self.nlp.obj(x[:nx])

		convals = self.get_infeas(x)

		alfunc += numpy.dot(pi,convals)
		alfunc += 0.5*rho*numpy.sum(convals**2)

		return alfunc
	# end def


	# Evaluate augmented Lagrangian gradient
	def grad(self, x, pi, rho, **kwargs):
		nlp = self.nlp 
		nx = self.nx
		nsLL_ind = nx + self.nsLL
		nsUU_ind = nsLL_ind + self.nsUU
		# nsLR_ind = nsUU_ind + self.nsLR
		# nsUR_ind = nsLR_ind + self.nsUR

		algrad = numpy.zeros(self.n,'d')
		algrad[:nx] = nlp.grad(x[:nx])

		convals = self.get_infeas(x)

		vec = pi + rho*convals
		if isinstance(nlp, MFModel):
			algrad[:nx] += nlp.jtprod(x[:nx],vec)
		else:
			algrad[:nx] += rho*numpy.dot(nlp.jac(x[:nx]).transpose(),vec)
		# end if

		algrad[nx:nsLL_ind] = -pi[nlp.lowerC] - rho*convals[nlp.lowerC]
		algrad[nsLL_ind:nsUU_ind] = pi[nlp.upperC] + rho*convals[nlp.upperC]
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


	def hprod(self, x, pi, rho, v, **kwargs):
		'''
		Compute the Hessian-vector produce of the Hessian of the augmented 
		Lagrangian with arbitrary vector v. Both exact and approximate 
		Hessians are supported.
		'''

		nlp = self.nlp
		nx = self.nx
		w = numpy.zeros(self.n,'d')

		# Non-slack variables
		if self.approxHess:
			# Approximate Hessian
			w[:nx] = self.Hessapp.matvec(v[:nx])
		else
			# Exact Hessian
			# Note: the code in this block has yet to be properly tested
			convals = self.get_infeas(x)
			w[:nx] = nlp.hprod(x[:nx],pi,v[:nx],**kwargs)
			for i in range(self.m):
				w[:nx] += rho*convals[i]*nlp.hiprod(i,x[:nx],v[:nx])
			# end for 
			if isinstance(nlp, MFModel):
				w[:nx] += rho*nlp.jtprod(x[:nx],nlp.jprod(x[:nx],v[:nx]))	
			else:
				J = nlp.jac
				w[:nx] += rho*numpy.dot(J.transpose(),numpy.dot(J,v[:nx]))
			# end if
		# end if 

		# Slack variables
		w[nx:] = rho*v[nx:]

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

	def __init__(self, nlp, rho_init=10., tau=10., **kwargs):

		'''
		Initialize augmented Lagrangian method and options.
		Any options that are not used here are passed on into the bound-
		constrained solver.
		'''

		self.alprob = AugmentedLagrangian(nlp,**kwargs)

		self.rho_init = rho_init
		self.tau = tau
		self.omega_init = rho_init**-1 # kwargs.get('omega_init',1.)
		self.eta_init = rho_init**-0.1 # kwargs.get('eta_init',0.5)
		self.a_omega = kwargs.get('a_omega',1.)
		self.b_omega = kwargs.get('b_omega',1.)
		self.a_eta = kwargs.get('a_eta',0.1)
		self.b_eta = kwargs.get('b_eta',0.9)
		self.omega_opt = kwargs.get('omega_opt',1.e-6)
		self.eta_opt = kwargs.get('eta_opt',1.e-6)

		# Maximum number of outer iterations (use maxiter or maxinner for TR)
		self.maxouter = kwargs.get('maxouter',100)

		# (Future: create a logger to track problem data and print)

        return

	# end def 


	def solve(self, **kwargs):

		'''
		Solve the optimization problem and return the solution.

		Currently, only equality constraints are supported in the optimization 
		problem formulation.
		'''

		# Initial data for optimization
		rho = self.rho_init
		x = self.alprob.x0
		pi = self.alprob.pi0

		# First function and gradient evaluation
		phi = self.alprob.obj(x,pi,rho)
		dphi = self.alprob.grad(x,pi,rho)

		Pdphi = self.alprob.project_gradient(x,dphi)
		Pmax = numpy.max(numpy.abs(Pdphi))
		max_cons = numpy.max(numpy.abs(self.alprob.get_infeas(x)))
		n_iter = 0

		omega = self.omega_init
		eta = self.eta_init

		# Convergence check
		converged = Pmax <= self.omega_opt and max_cons <= self.eta_opt

		# While not converged, loop
		while not converged and n_iter < self.maxouter:

			# Perform bound-constrained minimization 
			# *** Call to Trust-Region BQP class goes here ***
			bound_solver = BoundTR(self.alprob)
			bound_solver.solve()

			# Update penalty parameter or multipliers based on result 
			convals_new = self.alprob.get_infeas(x)
			max_cons_new = numpy.max(numpy.abs(convals_new))
            if max_cons_new <= eta:
            	# Update convergence check
            	dphi = self.alprob.grad(x,pi,rho)
            	Pdphi = self.alprob.project_gradient(x,dphi)
            	Pmax_new = numpy.max(numpy.abs(Pdphi))
            	if max_cons_new <= self.eta_opt and Pmax_new <= self.omega_opt:
            		converged = True 
            		break
            	# No change in rho, tighten tolerances
            	pi -= rho*convals_new
            	eta /= rho**self.b_eta
            	omega /= rho**self.b_omega
            else:
            	# Increase rho, reset tolerances based on new rho
            	rho *= tau
            	eta = rho**-self.a_eta
            	omega = rho**-self.a_omega
            # end if

            n_iter += 1

		# end while

		# Solution output, etc.

	# end def



# end class
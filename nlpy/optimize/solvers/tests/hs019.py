#!/usr/local/bin/python

'''
hs019.py

Solve problem 19 in the Hock-Schittkowski collection using the NLPy 
implementation of Lancelot.

'''

# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys
import pdb
import math
import copy
import time

# =============================================================================
# External Python modules
# =============================================================================
import numpy

# =============================================================================
# Extension modules
# =============================================================================
sys.path.append(os.path.abspath('~/git-code/nlpy'))
from nlpy.model.nlp import NLPModel
from nlpy.model.mfnlp import MFModel
from nlpy.optimize.solvers.auglag import AugmentedLagrangianFramework

# =============================================================================
# Functions for defining the minimization problem
# =============================================================================

def obj(x):

	f = (x[0] - 10.)**3 + (x[1] - 20.)**3

	return f

# end def 

def cons(x):

	c = numpy.zeros(2,'d')
	c[0] = (x[0] - 5.)**2 + (x[1] - 5.)**2 - 100.
	c[1] = -(x[1] - 5.)**2 - (x[0] - 6.)**2 + 82.81
	return c

# end def

def grad(x):

	g = numpy.zeros(2,'d')
	g[0] = 3*(x[0] - 10.)**2
	g[1] = 3*(x[1] - 20.)**2
	#print g
	return g

# end def 

def jac(x):

	J = numpy.zeros([2,2],'d')
	J[0,0] = 2*(x[0] - 5.)
	J[0,1] = 2*(x[1] - 5.)
	J[1,0] = -2*(x[0] - 6.)
	J[1,1] = -2*(x[1] - 5.)
	#print J
	return J

# end def

def jtprod(x,q):

	p = numpy.zeros(2,'d')
	p[0] = q[0]*(2*(x[0] - 5.)) + q[1]*(-2*(x[0] - 6.))
	p[1] = q[0]*(2*(x[1] - 5.)) + q[1]*(-2*(x[1] - 5.))
	return p

# end def

# =============================================================================
# Main program
# =============================================================================

n = 2
m = 2
x0 = numpy.array([20.1,5.84])
Lcon = numpy.zeros(2,'d')
Lvar = numpy.array([13.,0.])
Uvar = 100*numpy.ones(2,'d')

prob = NLPModel(n=n,m=m,name='HS019',x0=x0,Lcon=Lcon,Lvar=Lvar,Uvar=Uvar)
# prob = MFModel(n=n,m=m,name='HS019',x0=x0,Lcon=Lcon,Lvar=Lvar,Uvar=Uvar)
prob.obj = obj
prob.cons = cons
prob.grad = grad
prob.jac = jac
# prob.jtprod = jtprod

solver = AugmentedLagrangianFramework(prob, omega_opt=1.e-6, eta_opt=1.e-6,
    		magic_steps=True, approxHess=True, printlevel=2)
t0 = time.time()
solver.solve()
print 'Solved in %8.3f seconds.'%(time.time() - t0)

# x_trial = numpy.array([14.095,0.84296079,0.,0.])
# x_trial_2 = numpy.array([13,11,0.,0.])
# x_trial_3 = numpy.array([13,13,0.,0.])

# print solver.alprob.obj(x_trial)
# print solver.alprob.obj(x_trial_2)
# print solver.alprob.obj(x_trial_3)

# print solver.alprob.grad(x_trial)
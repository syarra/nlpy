#!/usr/local/bin/python

'''
hs029.py

Solve problem 29 in the Hock-Schittkowski collection using the NLPy 
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

	f = -x[0]*x[1]*x[2]

	return f

# end def 

def cons(x):

	c = numpy.zeros(1,'d')
	c[0] = -(x[0])**2 - 2*(x[1])**2 - 4*(x[2])**2 + 48.
	return c

# end def

def grad(x):

	g = numpy.zeros(3,'d')
	g[0] = -x[1]*x[2]
	g[1] = -x[0]*x[2]
	g[2] = -x[0]*x[1]
	return g

# end def 

def jac(x):

	J = numpy.zeros([1,3],'d')
	J[0,0] = -2*(x[0])
	J[0,1] = -4*(x[1])
	J[0,2] = -8*(x[2])
	return J

# end def

def jtprod(x,q):

	p = numpy.zeros(3,'d')
	p[0] = q[0]*(-2*x[0]) 
	p[1] = q[0]*(-4*x[1])
	p[2] = q[0]*(-8*x[2])
	return p

# end def

# =============================================================================
# Main program
# =============================================================================

n = 3
m = 1
x0 = numpy.ones(3,'d')
Lcon = numpy.zeros(1,'d')

prob = NLPModel(n=n,m=m,name='HS029',x0=x0,Lcon=Lcon)
# prob = MFModel(n=n,m=m,name='HS029',x0=x0,Lcon=Lcon)
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

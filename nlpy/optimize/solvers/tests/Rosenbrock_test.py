'''
Rosenbrock_test.py

Solve the unconstrained N-dimensional Rosenbrock problem using NLPy. 

This program is designed as a test of NLPy and an 
example use case.
'''

# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys
import pdb
import math
import copy

# =============================================================================
# External Python modules
# =============================================================================
import numpy as np
import numpy.random as random

# =============================================================================
# Extension modules
# =============================================================================
from nlpy.model.nlp import NLPModel
from nlpy.optimize.solvers.lbfgs import LBFGSFramework,LBFGS
from nlpy.optimize.solvers.ldfp import LDFP
from nlpy.optimize.solvers.lsr1 import LSR1
from nlpy.optimize.tr.trustregion import TrustRegionFramework, TrustRegionCG
from nlpy.optimize.solvers.trunk import LQNTrunkFramework

# =============================================================================
# Functions for defining the minimization problem
# =============================================================================

def obj(x):

	f = 0.
	for i in range(testprob.n - 1):
		f += 100*(x[i+1] - x[i]**2)**2 + (1. - x[i])**2
	# end for 

	return f

# end def 

def grad(x):

	g = np.zeros(testprob.n)
	g[0] = 200*(x[1] - x[0]**2)*(-2*x[0]) - 2*(1. - x[0])
	for i in range(1,testprob.n - 1):
		g[i] = 200*(x[i+1] - x[i]**2)*(-2*x[i]) - \
			2*(1. - x[i]) + 200*(x[i] - x[i-1]**2)
	# end for 
	g[testprob.n - 1] = 200*(x[testprob.n - 1] - x[testprob.n - 2]**2)
	return g

# end def 

# =============================================================================
# Main program
# =============================================================================

# Choose problem scale and starting point
n = 20
x0 = 2*np.ones(n)

testprob = NLPModel(n=n,m=0,name='Rosenbrock Test',x0=x0)
testprob.obj = obj
testprob.grad = grad

# Solve using a BFGS line search method
#usolver = LBFGSFramework(testprob,silent=True)
#usolver.solve()

# Solve using the TRUNK framework with a desired quasi-Newton approximation
# to the true Hessian
tr = TrustRegionFramework()
trsolver = TrustRegionCG
qn = LDFP
usolver2 = LQNTrunkFramework(testprob,tr,trsolver,QNApprox=qn)
usolver2.Solve()
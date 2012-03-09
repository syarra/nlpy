'''
scalable_AL_test.py

Form an NLPModel of a scalable test problem and verify that the conversion of 
the NLP or matrix-free NLP into an augmented Lagrangian problem proceeds 
correctly.
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
import numpy
import numpy.random

# =============================================================================
# Extension modules
# =============================================================================
from nlpy.model.nlp import NLPModel
from nlpy.model.mfnlp import MFModel
from nlpy.optimize.solvers.auglag import AugmentedLagrangian

# =============================================================================
# Functions for defining the minimization problem
# =============================================================================

def obj(x):

	f = 0.
	for i in range(testmodel.n):
		f += (2*i + 1)**0.5*x[i]**2
	# end for
	return f**0.5

# end def 

def cons(x):

	c = numpy.zeros(testmodel.m,'d')
	# Note: n == m
	for i in range(testmodel.m):
		c[i] = (2*i + 1)**0.5/(2*x[i]**2)
	# end for
	return c

# end def

def grad(x):

	g = numpy.zeros(testmodel.n,'d')
	sumterm = 0.
	for i in range(testmodel.n):
		g[i] = (2*i + 1)*x[i]
		sumterm += g[i]*x[i]
	# end for
	return g/sumterm**0.5

# end def 

def jac(x):

	J = numpy.zeros([testmodel.m,testmodel.n],'d')
	# Note: n == m
	for i in range(testmodel.n):
		J[i,i] = -(2*i + 1)**0.5/x[i]**3
	# end for
	return J

# end def 

def jprod(x,v):

	# Note: n == m
	for i in range(testmodel.n):
		v[i] *= -(2*i + 1)**0.5/x[i]**3
	# end for
	return v

# end def 

def jtprod(x,w):

	# J is square and symmetric for this problem, so ...
	return jprod(x,w)

# end def 

# =============================================================================
# Main program
# =============================================================================

n = 5
x0 = 3*numpy.ones(n,'d')
pi0 = numpy.zeros(n,'d')
Lvar = 1.e-4*numpy.ones(n,'d')
Ucon = numpy.ones(n,'d')

testmodel = NLPModel(n=n,m=n,name='Scalable Test 1',x0=x0,Lvar=Lvar,Ucon=Ucon)
testmodel.obj = obj
testmodel.cons = cons
testmodel.grad = grad
testmodel.jac = jac

testMFmodel = MFModel(n=n,m=n,name='Matrix-Free Scalable Test 1',x0=x0,
	Lvar=Lvar,Ucon=Ucon)
testMFmodel.obj = obj
testMFmodel.cons = cons
testMFmodel.grad = grad
testMFmodel.jprod = jprod
testMFmodel.jtprod = jtprod

# Test vectors for checking Jacobian vector product
vec1 = numpy.ones(n)
vec2 = numpy.arange(n,dtype='d')
vec3 = numpy.random.rand(n)

# Verify correctness of jprod and jtprod functions 
J = testmodel.jac(x0)
# The following vectors should all be zero
print 'Test Jacobian-vector product functions:'
print numpy.dot(J.transpose(),vec1) - testMFmodel.jtprod(x0,vec1)
print numpy.dot(J.transpose(),vec2) - testMFmodel.jtprod(x0,vec2)
print numpy.dot(J.transpose(),vec3) - testMFmodel.jtprod(x0,vec3)
print ''

testMFAL = AugmentedLagrangian(testMFmodel)
print 'Augmented Lagrangian model statistics:'
print 'Number of variables = ', testMFAL.n 
print 'Number of slacks = ', testMFAL.ns
print 'Starting point = ', testMFAL.x0
print 'Starting multipliers = ', testMFAL.pi0
print ''

rho = 2.
infeas = testMFAL.get_infeas(testMFAL.x0)
print 'Constraint infeasibility = ', infeas
print 'Current AL objective function = ', testMFAL.obj(testMFAL.x0,testMFAL.pi0,rho)
print 'Current AL gradient = ', testMFAL.grad(testMFAL.x0,testMFAL.pi0,rho)
print ''

print 'Expected AL objective:'
print testmodel.obj(x0) + numpy.dot(pi0,testMFAL.get_infeas(testMFAL.x0)) + 0.5*rho*numpy.sum(testMFAL.get_infeas(testMFAL.x0)**2)
print 'Expected AL gradient:'
ALgrad = numpy.zeros(testMFAL.n)
ALgrad[:testMFAL.nx] = testmodel.grad(x0)
ALgrad[:testMFAL.nx] += numpy.dot(J.transpose(),pi0)
ALgrad[:testMFAL.nx] += rho*numpy.dot(J.transpose(),infeas)
ALgrad[testMFAL.nx:] = pi0 + rho*infeas
print ALgrad
print ''

# Later: Incorporate an exact Hessian test case using the original model
# (i.e. not the matrix-free case)

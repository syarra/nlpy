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
import logging

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
from nlpy.optimize.solvers.auglag2 import AugmentedLagrangianLbfgsFramework, AugmentedLagrangianFramework
from nlpy.optimize.solvers.sbmin import SBMINLbfgsFramework, SBMINFramework
from nlpy.tools.logs import config_logger

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

def jprod(x,p):

	q = numpy.zeros(1,'d')
	q[0] = p[0]*(-2*(x[0])) + p[1]*(-4*(x[1])) + p[2]*(-8*(x[2]))
	return q

def jtprod(x,q):

	p = numpy.zeros(3,'d')
	p[0] = q[0]*(-2*x[0]) 
	p[1] = q[0]*(-4*x[1])
	p[2] = q[0]*(-8*x[2])
	return p

# end def

def hprod(x,u,v):

	w = numpy.zeros(3,'d')

	w[0] = -x[2]*v[1] - x[1]*v[2]
	w[1] = -x[2]*v[0] - x[0]*v[2]
	w[2] = -x[1]*v[0] - x[0]*v[1]

	w[0] -= -2*u[0]*v[0]
	w[1] -= -4*u[0]*v[1]
	w[2] -= -8*u[0]*v[2]

	return w

# =============================================================================
# Main program
# =============================================================================

n = 3
m = 1
x0 = numpy.ones(3,'d')
Lcon = numpy.zeros(1,'d')

# prob = NLPModel(n=n,m=m,name='HS029',x0=x0,Lcon=Lcon)
prob = MFModel(n=n,m=m,name='HS029',x0=x0,Lcon=Lcon)
prob.obj = obj
prob.cons = cons
prob.grad = grad
# prob.jac = jac
prob.jprod = jprod
prob.jtprod = jtprod
prob.hprod = hprod

fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
# hndlr = logging.FileHandler('hs029_ms.log',mode='w')
hndlr = logging.StreamHandler()
hndlr.setLevel(logging.INFO)
hndlr.setFormatter(fmt)

# Configure auglag logger.
auglaglogger = logging.getLogger('nlpy.auglag')
auglaglogger.setLevel(logging.INFO)
auglaglogger.addHandler(hndlr)
auglaglogger.propagate = False

# Configure sbmin logger.
sbminlogger = logging.getLogger('nlpy.sbmin')
sbminlogger.setLevel(logging.INFO)
sbminlogger.addHandler(hndlr)
sbminlogger.propagate = False

# Configure bqp logger.
# bqplogger = logging.getLogger('nlpy.bqp')
# bqplogger.setLevel(logging.DEBUG)
# bqplogger.addHandler(hndlr)
# bqplogger.propagate = False

# Configure lbfgs logger.
# lbfgslogger = logging.getLogger('nlpy.lbfgs')
# lbfgslogger.setLevel(logging.INFO)
# lbfgslogger.addHandler(hndlr)
# lbfgslogger.propagate = False

solver = AugmentedLagrangianFramework(prob, SBMINFramework, maxouter=50, printlevel=2)
# solver = AugmentedLagrangianLbfgsFramework(prob, SBMINLbfgsFramework, maxouter=50, printlevel=2)
t0 = time.time()
solver.solve()
print 'Solved in %8.3f seconds.'%(time.time() - t0)
print ' in %d' % solver.niter_total

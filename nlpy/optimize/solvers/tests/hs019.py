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

def jprod(x,q):

	p = numpy.zeros(2,'d')
	p[0] = q[0]*(2.*(x[0] - 5.)) + q[1]*(2.*(x[1] - 5.))
	p[1] = q[0]*(-2.*(x[0] - 6.)) + q[1]*(-2.*(x[1] - 5.))
	return p


def jtprod(x,q):

	p = numpy.zeros(2,'d')
	p[0] = q[0]*(2*(x[0] - 5.)) + q[1]*(-2*(x[0] - 6.))
	p[1] = q[0]*(2*(x[1] - 5.)) + q[1]*(-2*(x[1] - 5.))
	return p

# end def

def hprod(x,u,v):
    p = numpy.zeros(2,'d')
    p[0] = v[0]*6*(x[0]-10.)
    p[1] = v[1]*6*(x[1]-20.)

    p[0] -= 2.*u[0]*v[0]
    p[1] -= 2.*u[0]*v[1]

    p[0] -= -2.*u[1]*v[0]
    p[1] -= -2.*u[1]*v[1]

    return p

# =============================================================================
# Main program
# =============================================================================

n = 2
m = 2
x0 = numpy.array([20.1,5.84])
Lcon = numpy.zeros(2,'d')
Lvar = numpy.array([13.,0.])
Uvar = 100.*numpy.ones(2,'d')

#prob = NLPModel(n=n,m=m,name='HS019',x0=x0,Lcon=Lcon,Lvar=Lvar,Uvar=Uvar)
prob = MFModel(n=n,m=m,name='HS019',x0=x0,Lcon=Lcon,Lvar=Lvar,Uvar=Uvar)
prob.obj = obj
prob.cons = cons
prob.grad = grad
#prob.jac = jac
prob.jprod = jprod
prob.jtprod = jtprod
prob.hprod = hprod

fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler()
hndlr.setLevel(logging.DEBUG)
hndlr.setFormatter(fmt)

# Configure auglag logger.
auglaglogger = logging.getLogger('nlpy.auglag')
auglaglogger.setLevel(logging.DEBUG)
auglaglogger.addHandler(hndlr)
auglaglogger.propagate = False

# Configure sbmin logger.
sbminlogger = logging.getLogger('nlpy.sbmin')
sbminlogger.setLevel(logging.DEBUG)
sbminlogger.addHandler(hndlr)
sbminlogger.propagate = False

# Configure sbmin logger.
#bqplogger = logging.getLogger('nlpy.bqp')
#bqplogger.setLevel(logging.INFO)
#bqplogger.addHandler(hndlr)
#bqplogger.propagate = False

solver = AugmentedLagrangianFramework(prob, SBMINFramework, maxouter=50, magic_steps=False, printlevel=2, logger='essai', verbose=True)
t0 = time.time()
#solver = AugmentedLagrangianLbfgsFramework(prob, SBMINLbfgsFramework, maxouter=50, magic_steps=False, printlevel=2)

solver.solve()

print 'Solved in %8.3f seconds.'%(time.time() - t0)
print ' in %d' % solver.niter_total


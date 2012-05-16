#!/usr/local/bin/python

'''
infeas_qp.py

A simple (infeasible) quadratic problem to test the infeasibility detection 
method inside the augmented Lagrangian framework.

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
from nlpy.optimize.solvers.sbmin import SBMINLbfgsFramework, SBMINFramework
from nlpy.optimize.solvers.auglag2 import AugmentedLagrangianLbfgsFramework, AugmentedLagrangianFramework
from nlpy.tools.logs import config_logger

# =============================================================================
# Functions for defining the minimization problem
# =============================================================================

def obj(x):

    f = x[0]**2
    return f

# end def 

def cons(x):
    c = numpy.zeros(1,'d')
    c[0] = x[0]
    return c

def grad(x):

    g = numpy.zeros(1,'d')
    g[0] = 2*x[0]
    return g

# end def 

def jprod(x,p):
    q = numpy.zeros(1,'d')
    q[0] = 1.*p[0]
    return q

def jtprod(x,q):
    p = numpy.zeros(1,'d')
    p[0] = 1.*q[0]
    return p

def hprod(x,u,v):
    w = numpy.zeros(1,'d')
    w[0] = 2*v[0]
    return w

# =============================================================================
# Main program
# =============================================================================

n = 1
m = 1
x0 = numpy.array([-2.5])
Lvar = -1*numpy.ones(n,'d')
Uvar = 1*numpy.ones(n,'d')
Lcon = 2*numpy.ones(m,'d')
Ucon = 2*numpy.ones(m,'d')

prob = MFModel(n=n,m=m,name='HS038',x0=x0,Lvar=Lvar,Uvar=Uvar,
    Lcon=Lcon,Ucon=Ucon)
prob.obj = obj
prob.cons = cons
prob.grad = grad
prob.jprod = jprod
prob.jtprod = jtprod
prob.hprod = hprod

fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
# hndlr = logging.FileHandler('infeas_qp.log',mode='w')
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

# solver = AugmentedLagrangianFramework(prob, SBMINFramework)
solver = AugmentedLagrangianLbfgsFramework(prob, SBMINLbfgsFramework)
t0 = time.time()
solver.solve()
print 'Solved in %8.3f seconds.'%(time.time() - t0)
print ' in %d' % solver.niter_total

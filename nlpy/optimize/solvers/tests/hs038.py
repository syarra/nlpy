#!/usr/local/bin/python

'''
hs038.py

Solve problem 38 in the Hock-Schittkowski collection using the NLPy 
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
from nlpy.optimize.solvers.sbmin import SBMINLbfgsFramework, SBMINFramework
from nlpy.optimize.solvers.auglag2 import AugmentedLagrangianLbfgsFramework, AugmentedLagrangianFramework
from nlpy.tools.logs import config_logger

# =============================================================================
# Functions for defining the minimization problem
# =============================================================================

def obj(x):

    f = 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
    f += 90*(x[3] - x[2]**2)**2 + (1 - x[2])**2
    f += 10.1*((x[1] - 1)**2 + (x[3] - 1)**2)
    f += 19.8*(x[1] - 1)*(x[3] - 1)
    return f

# end def 

def cons(x):
    return numpy.zeros(0,'d')

def grad(x):

    g = numpy.zeros(4,'d')
    g[0] = 200*(x[1] - x[0]**2)*(-2*x[0]) - 2*(1 - x[0])
    g[1] = 200*(x[1] - x[0]**2) + 20.2*(x[1] - 1) + 19.8*(x[3] - 1)
    g[2] = 180*(x[3] - x[2]**2)*(-2*x[2]) - 2*(1 - x[2])
    g[3] = 180*(x[3] - x[2]**2) + 20.2*(x[3] - 1) + 19.8*(x[1] - 1)
    return g

# end def 

def jprod(x,p):
    return numpy.zeros(0,'d')

def jtprod(x,q):
    return numpy.zeros(4,'d')

def hprod(x,u,v):
    w = numpy.zeros(4,'d')

    w[0] = (2 - 400*(x[1] - x[0]**2) + 800*x[0]**2)*v[0] + (-400*x[0])*v[1]
    w[1] = (-400*x[0])*v[0] + (20.2 + 200.)*v[1] + (19.8)*v[3]
    w[2] = (2 - 360*(x[3] - x[2]**2) + 720*x[2]**2)*v[2] + (-360*x[2])*v[3]
    w[3] = (19.8)*v[1] + (-360*x[2])*v[2] + (180 + 20.2)*v[3]

    return w

# =============================================================================
# Main program
# =============================================================================

n = 4
m = 0
x0 = numpy.array([-3.,-1.,-3.,-1.])
Lvar = -10*numpy.ones(n,'d')
Uvar = 10*numpy.ones(n,'d')

prob = MFModel(n=n,m=m,name='HS038',x0=x0,Lvar=Lvar,Uvar=Uvar)
prob.obj = obj
prob.cons = cons
prob.grad = grad
prob.jprod = jprod
prob.jtprod = jtprod
prob.hprod = hprod

fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
# hndlr = logging.FileHandler('hs038.log',mode='w')
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

# solver = AugmentedLagrangianFramework(prob, SBMINFramework)
solver = AugmentedLagrangianLbfgsFramework(prob, SBMINLbfgsFramework)
t0 = time.time()
solver.solve(ny=True)
print 'Solved in %8.3f seconds.'%(time.time() - t0)
print ' in %d' % solver.niter_total

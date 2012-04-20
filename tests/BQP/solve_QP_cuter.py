"""
Python Script for solving a bunch of Ampl Model QP form the Cuter Collection
Sylvain Arreckx
"""

from nlpy.model import AmplModel
from nlpy.optimize.solvers.bqp import BQP
from pyOpt import IPOPT
from translatePyoptNLPy import *

import os.path

QPlist = open('QPlist.txt', 'r')


hdrfmt = '%-15s %5s %5s %15s %5s'
hdr = hdrfmt % ('Name','Dim','Iter','Objective','Opt')
lhdr = len(hdr)
fmt = '%-15s %5d %5d %15.8e %5s'
fmt_no = '%-15s'
print hdr
print '-'*lhdr

for ligne in QPlist.readlines():
    l = ligne.split()
    qpname = l[0].lower()
    ProblemName = '/Users/syarra/data/CuteExamples/nl_folder/'+qpname+'.nl'

    if os.path.isfile(ProblemName):
        nlp = AmplModel(ProblemName)
        bqp = BQP(nlp)
        bqp.solve(maxiter=10*nlp.n, stoptol=1.0e-8)
        print fmt % (qpname, nlp.n, bqp.niter, bqp.qval, repr(bqp.exitOptimal))

QPlist.close()

for file in ['ConvexQP1', 'ConvexQP2','ConvexQP3',
             'IndefiniteQP1','NonConvexQP1',]:

    nlp = AmplModel(file+'.nl')
    bqp = BQP(nlp)
    bqp.solve(maxiter=10*nlp.n, stoptol=1.0e-8)
    print fmt % (file, nlp.n, bqp.niter, bqp.qval, repr(bqp.exitOptimal))


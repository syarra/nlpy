from nlpy.model.mfnlp import MFAmplModel, SlackNLP
from nlpy.model.amplpy import AmplModel
from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.optimize.solvers import TrustBQPModel

import numpy as np
import sys



def FormEntireMatrix(on,om,Jop):
    J = np.zeros([om,on])
    for i in range(0,on):
        v = np.zeros(on)
        v[i] = 1.
        J[:,i] = Jop * v
    return J

for ProblemName in sys.argv[1:]:
    mfnlp = MFAmplModel(ProblemName)
    nlp = AmplModel(ProblemName)

    n = mfnlp.n
    print 'n = ', mfnlp.n
    print 'm = ', mfnlp.m

    x_k = mfnlp.x0

    print 'lvar : ', mfnlp.Lvar
    print 'uvar : ', mfnlp.Uvar

    print 'Orig gradient : ', mfnlp.grad(x_k)
    H = SimpleLinearOperator(n,n,symmetric=True,matvec=lambda u: mfnlp.hprod(x_k,None, u))

    print 'Orig hessian = \n', FormEntireMatrix(n,n,H)


    trbqp = TrustBQPModel(mfnlp, np.array([1.,-1.]), 1.)
    print 'x_k :', x_k

    x = np.array([-1.,-1])#np.ones(2)#trbqp.x0


    print 'lvar : ', trbqp.Lvar
    print 'uvar : ', trbqp.Uvar

    print 'Obj = ', trbqp.obj(x)
    print 'Gradient =', trbqp.grad(x)

    H = SimpleLinearOperator(n,n,symmetric=True,matvec=lambda u: trbqp.hprod(x,None, u))

    print 'H = \n', FormEntireMatrix(n,n,H)


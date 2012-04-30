from nlpy.model.mfnlp import MFAmplModel, SlackNLP
from nlpy.model.amplpy import AmplModel
from nlpy.optimize.solvers.auglag2 import AugmentedLagrangian

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

    on = mfnlp.n
    om = mfnlp.m

    print 'on = ', on
    print 'om = ', om

    ox = mfnlp.x0

    Jop = mfnlp.jac(ox)
    print 'J mf =\n', FormEntireMatrix(on,om,Jop)
    print 'J true =\n', nlp.jac(ox)


    Hop = mfnlp.hess(ox)
    print 'H mf =\n', FormEntireMatrix(on,on,Hop)
    print 'H true =\n', nlp.hess(ox)

    alprob = AugmentedLagrangian(mfnlp)

    n = alprob.n
    m = alprob.nlp.m

    x = alprob.x0
    alprob.pi = np.ones(alprob.nlp.m)

    Jop = alprob.nlp.jac(ox)
    print 'J mf =\n', FormEntireMatrix(n,m,Jop)

    Jop = alprob.nlp.jac(ox)
    print 'J mf =\n', FormEntireMatrix(m,n,Jop.T)


    print alprob.pi
    print 'original nlp cons : \n', nlp.cons(ox)
    print 'slackframework cons :\n', alprob.nlp.cons(x)

    print 'original nlp obj: ', nlp.obj(ox)
    print 'original nlp grad:\n', nlp.grad(ox)
    print 'grad =\n', alprob.grad(x)

    Hop = alprob.hess(x)
    print 'H mf =\n', FormEntireMatrix(n,n,Hop)

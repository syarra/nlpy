from nlpy.model.mfnlp import MFAmplModel, SlackNLP
from nlpy.model.amplpy import AmplModel
from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.model.slacks import SlackFramework

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

    slacknlp = SlackNLP(mfnlp, keep_variable_bounds=True)
    slfr = SlackFramework(ProblemName)

    n = slacknlp.n
    m = slacknlp.m

    x = np.zeros(n)
    x[:on] = ox.copy()

    slacknlp.display_basic_info()

    print 'slack nlp cons : \n', slacknlp.cons(x)
    print 'original nlp cons : \n', nlp.cons(x)
    print 'slackframework cons :\n', slfr.cons(x)

    Jop = slacknlp.jac(x)
    print 'J mf =\n', FormEntireMatrix(n,m,Jop)
    print 'J true =\n', nlp.jac(ox)
    print 'J slfr =\n', slfr.jac(x)

    Hop = mfnlp.hess(ox)
    print 'H mf =\n', FormEntireMatrix(on,on,Hop)
    print 'H true =\n', nlp.hess(ox)

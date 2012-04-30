from nlpy.model.mfnlp import MFAmplModel, SlackNLP
from nlpy.model.amplpy import AmplModel
import numpy as np
import sys

for ProblemName in sys.argv[1:]:
    mfnlp = MFAmplModel(ProblemName)
    nlp = AmplModel(ProblemName)

    print 'n = ', mfnlp.n
    print 'm = ', mfnlp.m

    x = np.ones(mfnlp.n)

    Jop = mfnlp.jac(x)

    J = np.zeros([1,2])
    J[:,0] = Jop * np.array([1.,0.])
    J[:,1] = Jop * np.array([0.,1.])

    print 'J mf =\n', J
    print 'J true =\n', nlp.jac(x)


    Hop = mfnlp.hess(x)
    H = np.zeros([2,2])
    H[:,0] = Hop * np.array([1.,0.])
    H[:,1] = Hop * np.array([0.,1.])

    print 'H mf =\n', H
    print 'H true =\n', nlp.hess(x)


    slacknlp = SlackNLP(mfnlp)

    Jop = slacknlp.jac(x)
    J = np.zeros([1,2])
    J[:,0] = Jop * np.array([1.,0.])
    J[:,1] = Jop * np.array([0.,1.])

    print 'J mf =\n', J
    print 'J true =\n', nlp.jac(x)

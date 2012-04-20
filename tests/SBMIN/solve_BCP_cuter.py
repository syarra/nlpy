"""
Python Script for solving a bunch of Ampl Model QP form the Cuter Collection
Sylvain Arreckx
"""

from nlpy.model import AmplModel
from nlpy.optimize.solvers.sbmin import SBMINFramework
from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
from nlpy.optimize.tr.trustregion import TrustRegionBQP as TRSolver
import os.path


ProblemList = open('BoxConstrained.txt', 'r')


hdrfmt = '%-10s %5s %5s %15s %5s'
hdr = hdrfmt % ('Name','Dim','Iter','Objective','Opt')
lhdr = len(hdr)
fmt = '%-10s %5d %5d %15.8e %5s'
fmt_no = '%-10s'
print hdr
print '-'*lhdr

for ligne in ProblemList.readlines():
    l = ligne.split()
    pbname = l[0].lower()
    ProblemName = '/Users/syarra/data/CuteExamples/nl_folder/'+pbname+'.nl'

    if os.path.isfile(ProblemName):
        nlp = AmplModel(ProblemName)
        if nlp.n <= 100:
            tr = TR(eta1=0.25, eta2=0.75, gamma1=0.0625, gamma2=2.)
            sbmin = SBMINFramework(nlp, tr, TRSolver,
                                   maxiter = 30*nlp.n, verbose=False)

            sbmin.Solve()
            print fmt % (pbname, nlp.n, sbmin.iter, sbmin.f, repr(sbmin.status))

    else: print fmt_no % pbname
ProblemList.close()


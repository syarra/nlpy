#!/usr/bin/env python

from nlpy.model.amplpy import MFAmplModel
from nlpy.optimize.solvers.sbmin import SBMINFramework
from nlpy.optimize.solvers.auglag2 import AugmentedLagrangianFramework
from nlpy.tools.timing import cputime
import numpy
import sys

def pass_to_auglag(nlp, showbanner=True):

    t = cputime()

    AUGLAG = AugmentedLagrangianFramework(nlp, SBMINFramework, maxouter=50)
    t_setup = cputime() - t                  # Setup time

    if showbanner:
        print
        print '------------------------------------------'
        print 'solving: Solving problem %-s' % ProblemName
        print '------------------------------------------'
        print

    AUGLAG.solve()

    # Output final statistics
    print
    print 'Final variables:', AUGLAG.x
    print
    print '-------------------------------'
    print 'LANCELOT: End of Execution'
    print '  Problem                               : %-s' % ProblemName
    print '  Dimension                             : %-d' % nlp.n
    print '  Initial/Final Objective               : %-g/%-g' % (AUGLAG.f0, AUGLAG.f)
    print '  Initial/Final Projected Gradient Norm : %-g/%-g' % (AUGLAG.pg0, AUGLAG.pgnorm)
    print '  Number of iterations        : %-d' % AUGLAG.niter_total
    print '-------------------------------'
    return AUGLAG

if len(sys.argv) < 2:
    sys.stderr.write('Please specify model name\n')
    sys.exit(-1)

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

for ProblemName in sys.argv[1:]:
    nlp = MFAmplModel(ProblemName)         # Create a model
    AUGLAG = pass_to_auglag(nlp, showbanner=True)
    nlp.close()                                 # Close connection with model

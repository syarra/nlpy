#!/usr/bin/env python

from nlpy.model import amplpy
from nlpy.optimize.solvers.auglag import AugmentedLagrangianFramework
from nlpy.tools.timing import cputime
import numpy
import sys

def pass_to_lancelot(nlp, showbanner=True):

    if nlp.nrangeC > 0:         # Check for range constrained problem
        sys.stderr.write('%s has %d general range constraints\n' % (ProblemName, nlp.nrangeC))
        return None

    t = cputime()

    LANCELOT = AugmentedLagrangianFramework(nlp, approxHess=True, printlevel=2)
    t_setup = cputime() - t                  # Setup time

    if showbanner:
        print
        print '------------------------------------------'
        print 'solving: Solving problem %-s' % ProblemName
        print '------------------------------------------'
        print

    LANCELOT.solve()

    # Output final statistics
    print
    print 'Final variables:', LANCELOT.x
    print
    print '-------------------------------'
    print 'LANCELOT: End of Execution'
    print '  Problem                               : %-s' % ProblemName
    print '  Dimension                             : %-d' % nlp.n
    print '  Initial/Final Objective               : %-g/%-g' % (LANCELOT.f0, LANCELOT.f)
    print '  Initial/Final Projected Gradient Norm : %-g/%-g' % (LANCELOT.pg0, LANCELOT.pgnorm)
    print '  Number of iterations        : %-d' % LANCELOT.iter
    print '-------------------------------'
    return LANCELOT

if len(sys.argv) < 2:
    sys.stderr.write('Please specify model name\n')
    sys.exit(-1)

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

for ProblemName in sys.argv[1:]:
    nlp = amplpy.AmplModel(ProblemName)         # Create a model
    print nlp.Uvar
    print nlp.Lvar
    LANCELOT = pass_to_lancelot(nlp, showbanner=True)
    nlp.close()                                 # Close connection with model

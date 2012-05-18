#!/usr/bin/env python

from nlpy.model.amplpy import MFAmplModel
from nlpy.optimize.solvers.sbmin import SBMINFramework
from nlpy.optimize.solvers.auglag2 import AugmentedLagrangianFramework
from nlpy.tools.timing import cputime
import numpy
import sys
import logging

def pass_to_auglag(nlp, showbanner=True, loglevel=2):

    t = cputime()

    AUGLAG = AugmentedLagrangianFramework(nlp, SBMINFramework, maxouter=50)
    t_setup = cputime() - t                  # Setup time

    if showbanner:
        print
        print '------------------------------------------'
        print 'solving: Solving problem %-s' % ProblemName
        print '------------------------------------------'
        print

    # Create loggers for studying output
    fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
    hndlr = logging.FileHandler('%-s_debug_%d.log' % (ProblemName,loglevel) ,mode='w')
    # hndlr = logging.StreamHandler()
    hndlr.setLevel(logging.DEBUG)
    hndlr.setFormatter(fmt)

    # Configure auglag logger.
    if loglevel >= 1:
        auglaglogger = logging.getLogger('nlpy.auglag')
        auglaglogger.setLevel(logging.DEBUG)
        auglaglogger.addHandler(hndlr)
        auglaglogger.propagate = False

    # Configure sbmin logger.
    if loglevel >= 2:
        sbminlogger = logging.getLogger('nlpy.sbmin')
        sbminlogger.setLevel(logging.DEBUG)
        sbminlogger.addHandler(hndlr)
        sbminlogger.propagate = False

    # Configure bqp logger.
    if loglevel >= 3:
        bqplogger = logging.getLogger('nlpy.bqp')
        bqplogger.setLevel(logging.DEBUG)
        bqplogger.addHandler(hndlr)
        bqplogger.propagate = False

    # Configure pcg logger
    if loglevel >= 4:
        pcglogger = logging.getLogger('nlpy.pcg')
        pcglogger.setLevel(logging.DEBUG)
        pcglogger.addHandler(hndlr)
        pcglogger.propagate = False

    AUGLAG.solve()

    t_solve = cputime() - t - t_setup

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
    print '  Solution time               : %-g s' % t_solve
    print '-------------------------------'
    return AUGLAG

if len(sys.argv) < 2:
    sys.stderr.write('Please specify model name\n')
    sys.exit(-1)

if len(sys.argv) < 3:
    sys.stderr.write('Please specify the logging level of detail\n')
    sys.exit(-1)

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

# for ProblemName in sys.argv[1:]:
#     nlp = MFAmplModel(ProblemName)         # Create a model
#     AUGLAG = pass_to_auglag(nlp, showbanner=True, loglevel=2)
#     nlp.close()                                 # Close connection with model

ProblemName = sys.argv[1]
loglevel = sys.argv[2]
nlp = MFAmplModel(ProblemName)         # Create a model
AUGLAG = pass_to_auglag(nlp, showbanner=True, loglevel=loglevel)
nlp.close()                                 # Close connection with model

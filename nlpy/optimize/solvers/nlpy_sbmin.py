#!/usr/bin/env python

from nlpy.model import amplpy
from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
from nlpy.optimize.tr.trustregion import TrustRegionBQP as TRSolver
from nlpy.optimize.solvers.sbmin import SBMINFramework
from nlpy.tools.timing import cputime
from nlpy.tools.logs import config_logger
import logging
import numpy
import sys

def pass_to_sbmin(nlp, showbanner=True):

    if nlp.m > 0:         # Check for constrained problem
        sys.stderr.write('%s has %d general constraints\n' % (ProblemName, nlp.m))
        return None

    t = cputime()
    tr = TR(eta1=0.25, eta2=0.75, gamma1=0.0625, gamma2=2)

    # we select a trust-region subproblem solver of our choice.
    SBMIN = SBMINFramework(nlp, tr, TRSolver, maxiter=100, verbose=True,
            ny=True, monotone=False, abstol=1.0e-8, reltol=1.0e-6)
    t_setup = cputime() - t                  # Setup time

    if showbanner:
        print
        print '------------------------------------------'
        print 'solving: Solving problem %-s with parameters' % ProblemName
        hdr = 'eta1 = %-g  eta2 = %-g  gamma1 = %-g  gamma2 = %-g Delta0 = %-g'
        print hdr % (tr.eta1, tr.eta2, tr.gamma1, tr.gamma2, tr.Delta)
        print '------------------------------------------'
        print

    SBMIN.Solve()

    # Output final statistics
    print
    print 'Final variables:', SBMIN.x
    print
    print '-------------------------------'
    print 'SBMIN: End of Execution'
    print '  Problem                               : %-s' % ProblemName
    print '  Dimension                             : %-d' % nlp.n
    print '  Initial/Final Objective               : %-g/%-g' % (SBMIN.f0, SBMIN.f)
    print '  Initial/Final Projected Gradient Norm : %-g/%-g' % (SBMIN.pg0, SBMIN.pgnorm)
    print '  Number of iterations        : %-d' % SBMIN.iter
    print '  Number of function evals    : %-d' % SBMIN.nlp.feval
    print '  Number of gradient evals    : %-d' % SBMIN.nlp.geval
    print '  Number of Hessian  evals    : %-d' % SBMIN.nlp.Heval
    print '  Number of matvec products   : %-d' % SBMIN.nlp.Hprod
    print '  Total/Average BQP iter      : %-d/%-g' % (SBMIN.total_bqpiter, (float(SBMIN.total_bqpiter)/SBMIN.iter))
    print '  Setup/Solve time            : %-gs/%-gs' % (t_setup, SBMIN.tsolve)
    print '  Total time                  : %-gs' % (t_setup + SBMIN.tsolve)
    print '  Status                      :', SBMIN.status
    print '-------------------------------'
    return SBMIN

if len(sys.argv) < 2:
    sys.stderr.write('Please specify model name\n')
    sys.exit(-1)

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

# Create root logger.
log = logging.getLogger('nlpy.sbmin')
level = logging.INFO
log.setLevel(level)
fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)

# Configure subproblem logger.
config_logger('nlpy.bqp',
              filename='sbmin-bqp.log',
              filemode='w',
              stream=None)

def apply_scaling(nlp):
    "Apply scaling to the NLP and print something if asked."
    gNorm = nlp.compute_scaling_obj()
    log.info('%17s: %8s %8s %8s'     % ('Scaling applied', '|g| unscaled', '|g| scaled', 'component'))
    log.info('%17s: %8.1e %8.1e'     % ('  objective', gNorm, nlp.scale_obj*gNorm))
    log.info('')

for ProblemName in sys.argv[1:]:
    nlp = amplpy.AmplModel(ProblemName)         # Create a model
    nlp.display_basic_info()
    apply_scaling(nlp)
    SBMIN = pass_to_sbmin(nlp, showbanner=True)
    nlp.close()                                 # Close connection with model


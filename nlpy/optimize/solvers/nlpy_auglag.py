#!/usr/bin/env python

from __future__ import with_statement # Required in 2.5
from nlpy import __version__
from nlpy.model import MFAmplModel
from nlpy.optimize.solvers.sbmin import SBMINFramework, SBMINLqnFramework
from nlpy.optimize.solvers.auglag2 import AugmentedLagrangianFramework, \
                                          AugmentedLagrangianLbfgsFramework, \
                                          AugmentedLagrangianLsr1Framework
from nlpy.tools.timing import cputime
from optparse import OptionParser
import numpy
import nlpy.tools.logs
import sys, logging, os
import threading
import subprocess
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException, "Timed out!"
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)



def pass_to_auglag(nlp, **kwargs):

    verbose = (kwargs['print_level'] == 2)
    qn = kwargs.get('quasi_newton',None)

    t = cputime()

    if qn == None:
        auglag = AugmentedLagrangianFramework(nlp, SBMINFramework,
                    **kwargs)
    elif qn == 'LBFGS':
        auglag = AugmentedLagrangianLbfgsFramework(nlp, SBMINLqnFramework,
                    **kwargs)
    elif qn == 'LSR1':
        auglag = AugmentedLagrangianLsr1Framework(nlp, SBMINLqnFramework,
                    **kwargs)

    t_setup = cputime() - t    # Setup time.

    try:
        with time_limit(kwargs['maxtime']):
            auglag.solve(**kwargs)
    except TimeoutException, msg:
        print "Timed out!"
        auglag.status=5
        auglag.tsolve=0
        auglag.cons_max=0
        auglag.pi_max=0

    return (t_setup, auglag)


usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent nonlinear programs."""

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

parser.add_option("-s", "--magic_steps", action="store", type="string",
                  default=None, dest="magic_steps",
                  help="Enable magical steps (None, 'cons': for conservative, or \
                        'agg': for aggressive)")
parser.add_option("-m", "--monotone", action="store_false",
                  default=True, dest="monotone",
                  help="Enable non monotone strategy")
parser.add_option("-y", "--nocedal_yuan", action="store_true",
                  default=False, dest="ny",
                  help="Enable Nocedal-Yuan backtracking strategy")
parser.add_option("-l", "--least_square_multipliers", action="store_true",
                  default=False, dest="least_squares_pi",
                  help="Enable least-square estimate of the Lagrange multipliers")
parser.add_option("-q", "--quasi_newton", action="store", type="string",
                  default=None, dest="quasi_newton",
                  help="Use quasi-newton approximation of Hessian \
                        (None, LBFGS, LSR1)")
parser.add_option("-i", "--iter", action="store", type="int", default=None,
                  dest="maxiter",  help="Specify maximum number of iterations")
parser.add_option("-t", "--time", action="store", type="int", default=1000,
                  dest="maxtime",  help="Specify maximum number of computation time")
parser.add_option("-p", "--print_level", action="store", type="int",
                  default=0, dest="print_level",
                  help="Print iterations detail (0, 1 or 2)")
parser.add_option("-o", "--output_file", action="store", type="string",
                  default=None, dest="output_file",
                  help="Redirect iterations detail in an output file")
# Parse command-line options
(options, args) = parser.parse_args()

# Translate options to input arguments.
opts = {}
if options.maxiter is not None:
    opts['maxiter'] = options.maxiter
if options.magic_steps is not None:
    if options.magic_steps == 'agg':
        opts['magic_steps_agg'] = True
    elif options.magic_steps == 'cons':
        opts['magic_steps_cons'] = True
opts['ny'] = options.ny
opts['least_squares_pi'] = options.least_squares_pi
opts['quasi_newton'] = options.quasi_newton
opts['monotone'] = options.monotone
opts['print_level'] = options.print_level
opts['maxtime'] = options.maxtime

 # Create root logger.
log = logging.getLogger('auglag2')
log.setLevel(logging.INFO)
fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')

if options.output_file is not None:
    opts['output_file'] = options.output_file
    hndlr = logging.FileHandler(options.output_file, mode='w')
else:
    hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

multiple_problems = len(args) > 1
error = False

if multiple_problems:
    # Define formats for output table.
    hdrfmt = '%-10s %5s %5s %5s %15s %8s %8s %8s %6s %8s'

    hdr = hdrfmt % ('Name','n','m','Iter','Objective', '||c||', '||pi||', '#J.Tv', 'Time', 'Exit code')
    lhdr = len(hdr)
    fmt = '%-10s %5d %5d %5d %15.8e %6.2e %6.2e %8d %6.2f %8d'
    log.info(hdr)
    log.info('-' * lhdr)
else:
    # Configure auglag logger.
    if options.print_level >= 1:
        auglaglogger = logging.getLogger('nlpy.auglag')
        auglaglogger.setLevel(logging.INFO)
        auglaglogger.addHandler(hndlr)
        auglaglogger.propagate = False

    # Configure sbmin logger.
    if options.print_level >= 2:
        sbminlogger = logging.getLogger('nlpy.sbmin')
        sbminlogger.setLevel(logging.INFO)
        sbminlogger.addHandler(hndlr)
        sbminlogger.propagate = False

    # Configure bqp logger.
    if options.print_level >= 3:
        bqplogger = logging.getLogger('nlpy.bqp')
        bqplogger.setLevel(logging.DEBUG)
        bqplogger.addHandler(hndlr)
        bqplogger.propagate = False

    # Configure pcg logger
    if options.print_level >= 4:
        pcglogger = logging.getLogger('nlpy.pcg')
        pcglogger.setLevel(logging.DEBUG)
        pcglogger.addHandler(hndlr)
        pcglogger.propagate = False 


# Solve each problem in turn.
for ProblemName in args:

    nlp = MFAmplModel(ProblemName)

    ProblemName = os.path.basename(ProblemName)
    if ProblemName[-3:] == '.nl':
        ProblemName = ProblemName[:-3]
    t = cputime()
    t_setup, auglag = pass_to_auglag(nlp, **opts)
    total_time = cputime() - t
    if multiple_problems:
        if nlp.m == 0:
            auglag.cons_max = auglag.pi_max = 0

        log.info(fmt % (ProblemName, nlp.n, nlp.m, auglag.niter_total, auglag.f,
                        auglag.cons_max, auglag.pi_max,
                        auglag.alprob.nlp.nlp.Hprod,
                        total_time, auglag.status))

    nlp.close()  # Close model.

if not multiple_problems and not error:
    # Output final statistics
    log.info('--------------------------------')
    log.info('Auglag: End of Execution')
    log.info('  Problem                      : %-s' % ProblemName)
    log.info('  Number of variables          : %-d' % nlp.n)
    log.info('  Number of linear constraints : %-d' % nlp.nlin)
    log.info('  Number of general constraints: %-d' % (nlp.m - nlp.nlin))
    log.info('  Initial/Final Objective      : %-g/%-g' % (auglag.f0, auglag.f))
    log.info('  Number of iterations         : %-d' % auglag.niter_total)
    log.info('  Number of function evals     : %-d' % auglag.alprob.nlp.nlp.feval)
    log.info('  Number of Jacobian matvecs   : %-d' % auglag.alprob.nlp.nlp.Jprod)
    log.info('  Number of Jacobian.T matvecs : %-d' % auglag.alprob.nlp.nlp.JTprod)
    log.info('  Number of Hessian matvecs    : %-d' % auglag.alprob.nlp.nlp.Hprod)
    log.info('  Setup/Solve time             : %-gs/%-gs' % (t_setup, auglag.tsolve))
    log.info('  Total time                   : %-gs' % (total_time))
    log.info('  Status                       : %-gs' % auglag.status)
    log.info('--------------------------------')

#!/usr/bin/env python

from __future__ import with_statement # Required in 2.5
from nlpy import __version__
from nlpy.model import MFAmplModel
from nlpy.optimize.solvers.sbmin import SBMINFramework, SBMINLqnFramework, SBMINPartialLqnFramework, \
                                        SBMINStructuredLqnFramework
from nlpy.optimize.solvers.auglag2 import AugmentedLagrangianFramework, \
                                          AugmentedLagrangianLbfgsFramework,\
                                          AugmentedLagrangianLsr1Framework, \
                                          AugmentedLagrangianPartialLbfgsFramework, \
                                          AugmentedLagrangianPartialLsr1Framework, \
                                          AugmentedLagrangianStructuredLbfgsFramework, \
                                          AugmentedLagrangianStructuredLsr1Framework
from nlpy.tools.timing import cputime
from nlpy.tools.logs import config_logger
from optparse import OptionParser
import numpy
import sys, logging, os
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

    qn = kwargs.get('quasi_newton',None)
    t = cputime()

    if qn == None:
        auglag = AugmentedLagrangianFramework(nlp, SBMINFramework,
                    **kwargs)
    elif qn == 'LBFGS':
        auglag = AugmentedLagrangianPartialLbfgsFramework(nlp, SBMINPartialLqnFramework,
                    **kwargs)
    elif qn == 'LSR1':
        auglag = AugmentedLagrangianPartialLsr1Framework(nlp, SBMINPartialLqnFramework,
                    **kwargs)
    elif qn == 'SLBFGS':
        auglag = AugmentedLagrangianStructuredLbfgsFramework(nlp, SBMINStructuredLqnFramework,
                    **kwargs)
    elif qn == 'SLSR1':
        auglag = AugmentedLagrangianStructuredLsr1Framework(nlp, SBMINStructuredLqnFramework,
                    **kwargs)

    t_setup = cputime() - t    # Setup time.

    try:
        with time_limit(kwargs['maxtime']):
            auglag.solve(**kwargs)
    except TimeoutException, msg:
        print "Timed out!"
        auglag.status=-5
        auglag.tsolve=0
        auglag.cons_max=0
        auglag.pi_max=0
        auglag.pgnorm=0


    return (t_setup, auglag)


usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent nonlinear programs."""

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

parser.add_option("-s", "--magic_steps", action="store", type="string",
                  default=None, dest="magic_steps",
                  help="Enable magical steps (None, 'cons': for conservative, or \
                        'agg': for aggressive)")
parser.add_option("-m", "--monotone", action="store_true",
                  default=False, dest="monotone",
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
parser.add_option("-Q", "--qn_pairs", action="store", type="int",
                  default=5, dest="qn_pairs",
                  help="Number of pairs used in the Quasi Newton Hessian approximation")
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
    opts['max_inner_iter'] = options.maxiter
if options.magic_steps is not None:
    if options.magic_steps == 'agg':
        opts['magic_steps_agg'] = True
    elif options.magic_steps == 'cons':
        opts['magic_steps_cons'] = True
opts['ny'] = options.ny
opts['least_squares_pi'] = options.least_squares_pi
opts['quasi_newton'] = options.quasi_newton
opts['qn_pairs'] = options.qn_pairs
opts['monotone'] = options.monotone
opts['print_level'] = options.print_level
opts['maxtime'] = options.maxtime

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

multiple_problems = len(args) > 1
error = False
fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')

# Create root logger.
nlpylogger = logging.getLogger('root')
nlpylogger.setLevel(logging.INFO)
if options.output_file == None:
    hndlr = logging.StreamHandler(sys.stdout)
else:
    hndlr = logging.FileHandler(options.output_file)
hndlr.setFormatter(fmt)
nlpylogger.addHandler(hndlr)

if multiple_problems:

    nlpylogger.propagate = False

    # Configure subproblem logger.
    config_logger('nlpy.auglag',
                filename='auglag.log',
                filemode='w',
                stream=None,
                propagate=False)

    # Define formats for output table.
    hdrfmt = '%-10s %5s %5s %8s %5s %15s %8s %8s %6s %8s'

    hdr = hdrfmt % ('Name', 'n', 'm', 'Hprod', 'Iter', 'Objective', '||c||',
                    '||pi||', 'Time', 'Exit code')
    lhdr = len(hdr)
    fmt = '%-10s %5d %5d %8d %5d %15.8e %6.2e %6.2e %6.2f %8d'
    nlpylogger.info(hdr)
    nlpylogger.info('-' * lhdr)

else:
    if options.output_file == None:
        hndlr = logging.StreamHandler(sys.stdout)
    else:
        hndlr = logging.FileHandler(options.output_file)
    hndlr.setFormatter(fmt)

    # Configure auglag logger.
    auglaglogger = logging.getLogger('nlpy.auglag')
    auglaglogger.addHandler(hndlr)
    auglaglogger.propagate = False
    auglaglogger.setLevel(logging.INFO)
    if options.print_level >= 5:
        auglaglogger.setLevel(logging.DEBUG)

    # Configure sbmin logger.
    if options.print_level >= 2:
        sbminlogger = logging.getLogger('nlpy.sbmin')
        sbminlogger.setLevel(logging.INFO)
        sbminlogger.addHandler(hndlr)
        sbminlogger.propagate = False
        if options.print_level >= 5:
            sbminlogger.setLevel(logging.DEBUG)


    # Configure bqp logger.
    if options.print_level >= 3:
        bqplogger = logging.getLogger('nlpy.bqp')
        bqplogger.setLevel(logging.DEBUG)
        bqplogger.addHandler(hndlr)
        bqplogger.propagate = False
        if options.print_level >= 5:
            bqplogger.setLevel(logging.DEBUG)


    # Configure pcg logger
    if options.print_level >= 4:
        pcglogger = logging.getLogger('nlpy.pcg')
        pcglogger.setLevel(logging.DEBUG)
        pcglogger.addHandler(hndlr)
        pcglogger.propagate = False
        if options.print_level >= 5:
            pcglogger.setLevel(logging.DEBUG)

def apply_scaling(nlp):
    gNorm = nlp.compute_scaling_obj()


# Solve each problem in turn.
for ProblemName in args:

    nlp = MFAmplModel(ProblemName)
    #apply_scaling(nlp)

    msg = 'You are trying to solve an unconstrained problem\n'
    msg += ' '*25+'with auglag2, you could have better results using\n'
    msg +=' '*25+'an unconstrained solver such as sbmin.'
    if nlp.m == 0:
        nlpylogger.warning(msg)

    t = cputime()
    t_setup, AUGLAG = pass_to_auglag(nlp, **opts)
    total_time = cputime() - t

    if multiple_problems:
        if nlp.m == 0:
            AUGLAG.pi_max = AUGLAG.cons_max = 0

        if AUGLAG.status != 0:
            AUGLAG.niter_total = - AUGLAG.niter_total
            total_time = -total_time

        problemName = os.path.splitext(os.path.basename(ProblemName))[0]
        nlpylogger.info(fmt % (problemName, nlp.n, nlp.m, AUGLAG.alprob.nlp.nlp.Hprod,
                        AUGLAG.niter_total, AUGLAG.f, AUGLAG.cons_max,
                        AUGLAG.pi_max, total_time, AUGLAG.status))

    nlp.close()  # Close model.

if not multiple_problems and not error:
    # Output final statistics
    auglaglogger.info('')
    auglaglogger.info('Final variables: %-s' % repr(AUGLAG.x))
    auglaglogger.info('')
    auglaglogger.info('--------------------------------')
    auglaglogger.info('Auglag: End of Execution')
    auglaglogger.info('  Problem                               : %-s' % ProblemName)
    auglaglogger.info('  Number of variables                   : %-d' % nlp.n)
    auglaglogger.info('  Number of linear constraints          : %-d' % nlp.nlin)
    auglaglogger.info('  Number of general constraints         : %-d' % (nlp.m - nlp.nlin))
    auglaglogger.info('  Initial/Final Objective               : %-g/%-g' % (AUGLAG.f0, AUGLAG.f))
    auglaglogger.info('  Initial/Final Projected Gradient Norm : %-g/%-g' % (AUGLAG.pg0, AUGLAG.pgnorm))
    auglaglogger.info('  Number of iterations         : %-d' % AUGLAG.niter_total)
    auglaglogger.info('  Number of function evals     : %-d' % AUGLAG.alprob.nlp.nlp.feval)
    auglaglogger.info('  Number of Jacobian matvecs   : %-d' % AUGLAG.alprob.nlp.nlp.Jprod)
    auglaglogger.info('  Number of Jacobian.T matvecs : %-d' % AUGLAG.alprob.nlp.nlp.JTprod)
    auglaglogger.info('  Number of Hessian matvecs    : %-d' % AUGLAG.alprob.nlp.nlp.Hprod)
    auglaglogger.info('  Setup/Solve time             : %-gs/%-gs' % (t_setup, AUGLAG.tsolve))
    auglaglogger.info('  Total time                   : %-gs' % (total_time))
    auglaglogger.info('  Status                       : %-g' % AUGLAG.status)
    auglaglogger.info('--------------------------------')

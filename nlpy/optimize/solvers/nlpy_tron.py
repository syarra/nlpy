#!/usr/bin/env python

from nlpy import __version__
from nlpy.model import AmplModel
from nlpy.optimize.solvers.tron import TronFramework, TronLqnFramework
from nlpy.tools.timing import cputime
from optparse import OptionParser
import numpy
from nlpy.tools.logs import config_logger
import sys, logging, os


def pass_to_tron(nlp, **kwargs):
    qn = kwargs.get('quasi_newton', None)

    t = cputime()
    if qn == None:
        tron = TronFramework(nlp, **kwargs)
    else:
        tron = TronLqnFramework(nlp, **kwargs)

    t_setup = cputime() - t    # Setup time.

    tron.Solve()

    return (t_setup, tron)


usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent unconstrained nonlinear programs."""

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

parser.add_option("--reltol", action="store", type="float",
                  default=1.0e-6, dest="reltol",
                  help="Relative stopping tolerance")
parser.add_option("--abstol", action="store", type="float",
                  default=1.0e-6, dest="abstol",
                  help="Absolute stopping tolerance")
parser.add_option("-i", "--iter", action="store", type="int", default=None,
                  dest="maxit",  help="Specify maximum number of iterations")
parser.add_option("-q", "--quasi_newton", action="store", type="string",
                  default=None, dest="quasi_newton",
                  help="LBFGS or LSR1")


#
# Parse command-line options
(options, args) = parser.parse_args()

# Translate options to input arguments.
opts = {}
if options.maxit is not None:
    opts['maxit'] = options.maxit
opts['reltol'] = options.reltol
opts['abstol'] = options.abstol
opts['quasi_newton'] = options.quasi_newton


# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

multiple_problems = len(args) > 1
error = False

# Create root logger.
log = logging.getLogger('root')
log.setLevel(logging.INFO)
fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)


if multiple_problems:
    log.propagate = False

    # Configure subproblem logger.
    config_logger('nlpy.tron',
                filename='tron.log',
                filemode='w',
                stream=None,
                propagate=False)

   # Define formats for output table.
    hdrfmt = '%-10s %5s %5s %5s %15s %6s %6s %6s'
    hdr = hdrfmt % ('Name','n', 'Iter', 'Hprod', 'Objective',
                    'Setup','Solve','Status')
    lhdr = len(hdr)
    fmt = '%-10s %5d %5d %5d %15.8e %6.2f %6.2f %6s'
    log.info(hdr)
    log.info('-' * lhdr)
else:
    hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)

    # Configure the solver logger.
    sublogger = logging.getLogger('nlpy.tron')
    sublogger.setLevel(logging.INFO)
    sublogger.addHandler(hndlr)
    sublogger.propagate = False

def apply_scaling(nlp):
    "Apply scaling to the NLP and print something if asked."
    gNorm = nlp.compute_scaling_obj()

# Solve each problem in turn.
for ProblemName in args:

    nlp = AmplModel(ProblemName)
    ProblemName = os.path.splitext(os.path.basename(ProblemName))[0]
    apply_scaling(nlp)

    # Check for inequality- or equality-constrained problem.
    if nlp.m > 0:
        msg = '%s has %d linear or nonlinear constraints\n'
        log.error(msg % (ProblemName, nlp.m))
        error = True
    else:
        t_setup, tron = pass_to_tron(nlp, **opts)


    if multiple_problems and not error:
        log.info(fmt % (ProblemName, nlp.n, tron.iter, tron.nlp.Hprod,
                        tron.f, t_setup, t_setup+tron.tsolve, tron.status))

    nlp.close()  # Close model.


if not multiple_problems and not error:
    # Output final statistics
    log.info('--------------------------------')
    log.info('Tron: End of Execution')
    log.info('  Problem                      : %-s' % ProblemName)
    log.info('  Number of variables          : %-d' % nlp.n)
    log.info('  Initial/Final Objective      : %-g/%-g' % (tron.f0, tron.f))
    log.info('  Initial/Final Projected Gradient Norm : %-g/%-g' % (tron.gpnorm0, tron.gpnorm))
    log.info('  Number of iterations         : %-d' % tron.iter)
    log.info('  Number of function evals     : %-d' % tron.nlp.feval)
    log.info('  Number of gradient evals     : %-d' % tron.nlp.geval)
    #log.info('  Number of Hessian  evals     : %-d' % tron.nlp.Heval)
    log.info('  Number of Hessian matvecs    : %-d' % tron.nlp.Hprod)
    log.info('  Setup/Solve time             : %-gs/%-gs' % (t_setup, tron.tsolve))
    log.info('  Total time                   : %-gs' % (t_setup + tron.tsolve))
    log.info('  Status                       : %-s' % tron.status)
    log.info('--------------------------------')

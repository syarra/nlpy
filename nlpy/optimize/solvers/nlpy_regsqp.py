#!/usr/bin/env python

from nlpy import __version__
from nlpy.model import MFAmplModel, AmplModel
from nlpy.optimize.solvers.regsqp import RegSQPSolver, RegSQPBFGSIterativeSolver
from nlpy.tools.timing import cputime
from optparse import OptionParser
import numpy
from pykrylov.lls import LSMRFramework
import nlpy.tools.logs
import sys, logging, os

# Create root logger.
log = logging.getLogger('regsqp')
log.setLevel(logging.INFO)
fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
log.addHandler(hndlr)
log.propagate = False

# Configure the solver logger.
sublogger = logging.getLogger('regsqp.solver')
sublogger.setLevel(logging.INFO)
sublogger.addHandler(hndlr)
sublogger.propagate = False


def pass_to_regsqp(nlp, **kwargs):

    qn = kwargs.pop('quasi_newton')

    t = cputime()
    if qn:
        # This will need to be fixed when the quasi newton solver will be ready
        regsqpsolver = RegSQPBFGSIterativeSolver(nlp, LSMRFramework, **kwargs)
    else:
        regsqpsolver = RegSQPSolver(nlp, **kwargs)
    t_setup = cputime() - t    # Setup time.

    regsqpsolver.solve()
    return (t_setup, regsqpsolver)


usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent equality constrained nonlinear programs."""

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

parser.add_option("-a", "--abstol", action="store", type="float",
                  default=1.0e-6, dest="abstol",
                  help="Absolute stopping tolerance")
parser.add_option("-r", "--reltol", action="store", type="float",
                  default=1.0e-8, dest="reltol",
                  help="Absolute stopping tolerance")
parser.add_option("-t", "--theta", action="store", type="float",
                  default=0.99, dest="theta",
                  help="Sufficient decrease condition for the inner iterations")
parser.add_option("-d", "--direct", action="store_true",
                  default=False, dest="direct",
                  help="Direct or iterative solver")
parser.add_option("-q", "--quasi_newton", action="store_true",
                  default=False, dest="quasi_newton",
                  help="Use BFGS approximation of Hessian")
parser.add_option("-i", "--iter", action="store", type="int", default=None,
                  dest="maxiter",  help="Specify maximum number of iterations")
parser.add_option("-o", "--print_level", action="store", type="int",
                  default=0, dest="print_level",
                  help="Print iterations detail (0, 1 or 2)")

# Parse command-line options
(options, args) = parser.parse_args()

# Translate options to input arguments.
opts = {}
if options.maxiter is not None:
    opts['maxiter'] = options.maxiter
opts['abstol'] = options.abstol
opts['reltol'] = options.reltol
opts['theta'] = options.theta
opts['direct'] = options.direct
opts['quasi_newton'] = options.quasi_newton
opts['print_level'] = options.print_level

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

multiple_problems = len(args) > 1
error = False

if multiple_problems:
    # Define formats for output table.
    hdrfmt = '%-10s %5s %5s %15s %15s %6s %6s %5s'
    hdr = hdrfmt % ('Name','Iter','Feval','Objective','|| c ||',
                    'Setup','Solve','Opt')
    lhdr = len(hdr)
    fmt = '%-10s %5d %5d %15.8e %15.8e %6.2f %6.2f %5s'
    log.info(hdr)
    log.info('-' * lhdr)

# Solve each problem in turn.
for ProblemName in args:

    if options.quasi_newton:
      nlp = MFAmplModel(ProblemName)
    else:
      nlp = AmplModel(ProblemName)

    # Check for equality-constrained problem.
    n_ineq = nlp.nlowerC + nlp.nupperC + nlp.nrangeC
    if nlp.nbounds > 0 or n_ineq > 0:
        msg = '%s has %d bounds and %d inequality constraints\n'
        log.error(msg % (nlp.name, nlp.nbounds, n_ineq))
        error = True
    else:
        ProblemName = os.path.basename(ProblemName)
        if ProblemName[-3:] == '.nl':
            ProblemName = ProblemName[:-3]
        t_setup, regsqpsolver = pass_to_regsqp(nlp, **opts)
        if multiple_problems:
            log.info(fmt % (ProblemName, regsqpsolver.iter, regsqpsolver.nlp.feval, regsqpsolver.f,
                                    regsqpsolver.cnorm, t_setup, regsqpsolver.solve_time, regsqpsolver.optimal))
    nlp.close()  # Close model.

if not multiple_problems and not error:
    # Output final statistics
    log.info('--------------------------------')
    log.info('regsqp: End of Execution')
    log.info('  Problem                      : %-s' % ProblemName)
    log.info('  Number of variables          : %-d' % nlp.n)
    log.info('  Number of linear constraints : %-d' % nlp.nlin)
    log.info('  Number of general constraints: %-d' % (nlp.m - nlp.nlin))
    log.info('  Initial/Final Objective      : %-g/%-g' % (regsqpsolver.f0, regsqpsolver.f))
    log.info('  Number of iterations         : %-d' % regsqpsolver.iter)
    log.info('  Number of function evals     : %-d' % regsqpsolver.nlp.feval)
    log.info('  Number of gradient evals     : %-d' % regsqpsolver.nlp.geval)
    log.info('  Setup/Solve time             : %-gs/%-gs' % (t_setup, regsqpsolver.solve_time))
    log.info('  Total time                   : %-gs' % (t_setup + regsqpsolver.solve_time))
    log.info('--------------------------------')

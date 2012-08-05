#!/usr/bin/env python

from nlpy import __version__
from nlpy.model import amplpy
from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
from nlpy.optimize.tr.trustregion import TrustRegionBQP as TRSolver
from nlpy.optimize.solvers.sbmin import SBMINFramework
from nlpy.tools.timing import cputime
from nlpy.tools.logs import config_logger
import logging
from optparse import OptionParser
import numpy
import sys


def pass_to_sbmin(nlp, **kwargs):

    if nlp.m > 0:         # Check for constrained problem
        sys.stderr.write('%s has %d general constraints\n' % (ProblemName, nlp.m))
        return None

    t = cputime()
    tr = TR(eta1=0.25, eta2=0.75, gamma1=0.0625, gamma2=2)

    # we select a trust-region subproblem solver of our choice.
    sbmin = SBMINFramework(nlp, tr, TRSolver, **kwargs)

    t_setup = cputime() - t                  # Setup time

    sbmin.Solve()

    return (t_setup, sbmin)

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
parser.add_option("-i", "--iter", action="store", type="int", default=None,
                  dest="maxit",  help="Specify maximum number of iterations")
parser.add_option("-p", "--print_level", action="store", type="int",
                  default=0, dest="print_level",
                  help="Print iterations detail (0, 1 or 2)")
parser.add_option("-r", "--plot_radi", action="store_true",
                  default=False, dest="plot_radi",
                  help="Plot the evolution of the trust-region radius")
parser.add_option("-o", "--output_file", action="store", type="string",
                  default=None, dest="output_file",
                  help="Redirect iterations detail in an output file")
# Parse command-line options
(options, args) = parser.parse_args()

# Translate options to input arguments.
opts = {}
if options.maxit is not None:
    opts['maxiter'] = options.maxit
if options.magic_steps is not None:
    if options.magic_steps == 'agg':
        opts['magic_steps_agg'] = True
    elif options.magic_steps == 'cons':
        opts['magic_steps_cons'] = True
opts['ny'] = options.ny
opts['monotone'] = options.monotone
opts['print_level'] = options.print_level
opts['plot_radi'] = options.plot_radi

# Create root logger.
log = logging.getLogger('nlpy.sbmin')
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

# Configure subproblem logger.
config_logger('nlpy.bqp',
              filename='sbmin-bqp.log',
              filemode='w',
              stream=None)

def apply_scaling(nlp):
    "Apply scaling to the NLP and print something if asked."
    gNorm = nlp.compute_scaling_obj()
    log.info('%17s: %8s %8s %8s'     % ('Scaling applied', '|g| unscaled', '|g| scaled', 'component'))
    log.info('%17s: %8.1e %8.1e'     % ('  objective', gNorm, nlp.scale_obj * gNorm))
    log.info('')

multiple_problems = len(args) > 1
error = False

if multiple_problems:
    # Define formats for output table.
    hdrfmt = '%-10s %5s %5s %5s %15s %6s %5s'

    hdr = hdrfmt % ('Name', 'n', 'm', 'Iter', 'Objective', 'Solve', 'Opt')
    lhdr = len(hdr)
    fmt = '%-10s %5d %5d %5d %15.8e %5s %6.2f'
    log.info(hdr)
    log.info('-' * lhdr)
else:
    # Configure sbmin logger.
    if options.print_level >= 1:
        sbminlogger = logging.getLogger('nlpy.sbmin')
        sbminlogger.setLevel(logging.DEBUG)
        sbminlogger.addHandler(hndlr)
        sbminlogger.propagate = False

    # Configure bqp logger.
    if options.print_level >= 2:
        bqplogger = logging.getLogger('nlpy.bqp')
        bqplogger.setLevel(logging.DEBUG)
        bqplogger.addHandler(hndlr)
        bqplogger.propagate = False

    # Configure pcg logger
    if options.print_level >= 3:
        pcglogger = logging.getLogger('nlpy.pcg')
        pcglogger.setLevel(logging.DEBUG)
        pcglogger.addHandler(hndlr)
        pcglogger.propagate = False

# Solve each problem in turn.
for ProblemName in args:
    nlp = amplpy.MFAmplModel(ProblemName)         # Create a model
    apply_scaling(nlp)

    t = cputime()
    t_setup, SBMIN = pass_to_sbmin(nlp, **opts)
    total_time = cputime() - t

    if multiple_problems:
        log.info(fmt % (ProblemName, nlp.n, nlp.m, SBMIN.iter, SBMIN.f,
                        repr(SBMIN.status), total_time))

    nlp.close()                                 # Close connection with model

if not multiple_problems and not error:
    # Output final statistics
    log.info('')
    log.info('Final variables: %-s' % repr(SBMIN.x))
    log.info('')
    log.info('-------------------------------')
    log.info('SBMIN: End of Execution')
    log.info('  Problem                               : %-s' % ProblemName)
    log.info('  Dimension                             : %-d' % nlp.n)
    log.info('  Initial/Final Objective               : %-g/%-g' % (SBMIN.f0, SBMIN.f))
    log.info('  Initial/Final Projected Gradient Norm : %-g/%-g' % (SBMIN.pg0, SBMIN.pgnorm))
    log.info('  Number of iterations        : %-d' % SBMIN.iter)
    log.info('  Number of function evals    : %-d' % SBMIN.nlp.feval)
    log.info('  Number of gradient evals    : %-d' % SBMIN.nlp.geval)
    log.info('  Number of Hessian  evals    : %-d' % SBMIN.nlp.Heval)
    log.info('  Number of Jacobian matvecs   : %-d' % SBMIN.nlp.Jprod)
    log.info('  Number of Jacobian.T matvecs : %-d' % SBMIN.nlp.JTprod)
    log.info('  Number of Hessian matvecs    : %-d' % SBMIN.nlp.Hprod)
    log.info('  Total/Average BQP iter      : %-d/%-g' % (SBMIN.total_bqpiter, (float(SBMIN.total_bqpiter)/SBMIN.iter)))
    log.info('  Setup/Solve time            : %-gs/%-gs' % (t_setup, SBMIN.tsolve))
    log.info('  Total time                  : %-gs' % (total_time))
    log.info('  Status                      : %-s', SBMIN.status)
    log.info('-------------------------------')

    # Plot the evolution of the trust-region radius on the last problem
    if opts['plot_radi'] == True:
        try:
            import pylab
        except:
            sys.stderr.write('If you had pylab installed, you would be looking ')
            sys.stderr.write('at a plot of the evolution of the trust-region ')
            sys.stderr.write('radius, right now.\n')
            sys.exit(0)
        radii = numpy.array(SBMIN.radii, 'd')
        #pylab.plot(numpy.where(radii < 100, radii, 100))
        pylab.plot(radii)
        pylab.title('Trust-region radius')
        pylab.show()

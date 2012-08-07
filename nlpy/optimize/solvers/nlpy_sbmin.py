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
    sbmin = SBMINFramework(nlp, tr, TRSolver, **kwargs)
    t_setup = cputime() - t                  # Setup time
    sbmin.Solve()
    return (t_setup, sbmin)


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

# Set printing standards for arrays
numpy.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

multiple_problems = len(args) > 1
error = False
fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')

if multiple_problems:

    # Create root logger.
    sbminlogger = logging.getLogger('nlpy.sbmin')
    sbminlogger.setLevel(logging.INFO)

    if options.output_file is not None:
        opts['output_file'] = options.output_file
        hndlr = logging.FileHandler(options.output_file, mode='w')
    else:
        hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)
    sbminlogger.addHandler(hndlr)

    # Configure subproblem logger.
    config_logger('nlpy.bqp',
                filename='sbmin-bqp.log',
                filemode='w',
                stream=None)

    # Define formats for output table.
    hdrfmt = '%-10s %5s %5s %5s %15s %6s %6s'

    hdr = hdrfmt % ('Name', 'n', 'Iter', 'Hprod', 'Objective', 'Solve', 'Opt')
    lhdr = len(hdr)
    fmt = '%-10s %5d %5d %5d %15.8e %6s %6.2f'
    sbminlogger.info(hdr)
    sbminlogger.info('-' * lhdr)

else:

    hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)

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


def apply_scaling(nlp):
    "Apply scaling to the NLP and print something if asked."
    gNorm = nlp.compute_scaling_obj()
    sbminlogger.info('%17s: %8s %8s' % ('Scaling applied', 'g unscaled', '|g| scaled'))
    sbminlogger.info('%17s: %8.1e %8.1e'     % ('  objective', gNorm, nlp.scale_obj * gNorm))
    sbminlogger.info('')

# Solve each problem in turn.
for ProblemName in args:
    nlp = amplpy.MFAmplModel(ProblemName)         # Create a model
    apply_scaling(nlp)

    t = cputime()
    t_setup, SBMIN = pass_to_sbmin(nlp, **opts)
    total_time = cputime() - t

    if multiple_problems:
        sbminlogger.info(fmt % (ProblemName, nlp.n, SBMIN.iter, SBMIN.nlp.Hprod,
                                SBMIN.f, repr(SBMIN.status), total_time))

    nlp.close()                                 # Close connection with model

if not multiple_problems and not error:
    # Output final statistics
    sbminlogger.info('')
    sbminlogger.info('Final variables: %-s' % repr(SBMIN.x))
    sbminlogger.info('')
    sbminlogger.info('-------------------------------')
    sbminlogger.info('SBMIN: End of Execution')
    sbminlogger.info('  Problem                               : %-s' % ProblemName)
    sbminlogger.info('  Dimension                             : %-d' % nlp.n)
    sbminlogger.info('  Initial/Final Objective               : %-g/%-g' % (SBMIN.f0, SBMIN.f))
    sbminlogger.info('  Initial/Final Projected Gradient Norm : %-g/%-g' % (SBMIN.pg0, SBMIN.pgnorm))
    sbminlogger.info('  Number of iterations        : %-d' % SBMIN.iter)
    sbminlogger.info('  Number of function evals    : %-d' % SBMIN.nlp.feval)
    sbminlogger.info('  Number of gradient evals    : %-d' % SBMIN.nlp.geval)
    sbminlogger.info('  Number of Hessian  evals    : %-d' % SBMIN.nlp.Heval)
    sbminlogger.info('  Number of Hessian matvecs   : %-d' % SBMIN.nlp.Hprod)
    sbminlogger.info('  Total/Average BQP iter      : %-d/%-g' % (SBMIN.total_bqpiter, (float(SBMIN.total_bqpiter)/SBMIN.iter)))
    sbminlogger.info('  Setup/Solve time            : %-gs/%-gs' % (t_setup, SBMIN.tsolve))
    sbminlogger.info('  Total time                  : %-gs' % (total_time))
    sbminlogger.info('  Status                      : %-s', SBMIN.status)
    sbminlogger.info('-------------------------------')

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
        pylab.plot(radii)
        pylab.title('Trust-region radius')
        pylab.show()

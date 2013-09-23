#!/usr/bin/env python

from nlpy import __version__
from nlpy.model import amplpy
from nlpy.optimize.solvers.algencan import AlgencanFramework, AlgencanModel
from nlpy.tools.timing import cputime
from nlpy.tools.logs import config_logger
import logging
from optparse import OptionParser
import numpy as np
import sys
import os


def pass_to_algencan(nlp, **kwargs):

    try: 
        algencan = AlgencanFramework(nlp, **kwargs)
        algencan.Solve()
    except:
        print 'failed to solve'
    return algencan


if len(sys.argv) < 2:
    sys.stderr.write('Please specify model name\n')
    sys.exit(-1)

usage_msg = """%prog [options] problem1 [... problemN]
where problem1 through problemN represent nonlinear programs."""

# Define allowed command-line options
parser = OptionParser(usage=usage_msg, version='%prog version ' + __version__)

parser.add_option("-m", "--monotone", action="store_true",
                  default=False, dest="monotone",
                  help="Enable monotone strategy")
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
parser.add_option("-q", "--quasi_newton", action="store", type="string",
                  default=None, dest="quasi_newton",
                  help="LBFGS or LSR1")


# Parse command-line options:
(options, args) = parser.parse_args()

# Translate options to input arguments.
opts = {}
if options.maxit is not None:
    opts['maxiter'] = options.maxit
opts['ny'] = options.ny
opts['monotone'] = options.monotone
opts['print_level'] = options.print_level
opts['plot_radi'] = options.plot_radi
opts['quasi_newton'] = options.quasi_newton

# Set printing standards for arrays
np.set_printoptions(precision=3, linewidth=80, threshold=10, edgeitems=3)

multiple_problems = len(args) > 1
error = False
fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')

# Create root logger.
nlpylogger = logging.getLogger('root')
nlpylogger.setLevel(logging.INFO)
hndlr = logging.StreamHandler(sys.stdout)
hndlr.setFormatter(fmt)
nlpylogger.addHandler(hndlr)

if multiple_problems:

    nlpylogger.propagate = False

    # Configure subproblem logger.
    config_logger('nlpy.algencan',
                filename='algencan.log',
                filemode='w',
                stream=None,
                propagate=False)

    # Define formats for output table.
    hdrfmt = '%-10s %5s %5s %5s %15s %6s %6s'

    hdr = hdrfmt % ('Name', 'n', 'Iter', 'Hprod', 'Objective', 'Solve', 'Time')
    lhdr = len(hdr)
    fmt = '%-10s %5d %5d %5d %15.8e %6s %6.2f'
    nlpylogger.info(hdr)
    nlpylogger.info('-' * lhdr)

else:

    hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)

    # Configure algencan logger.
    algencanlogger = logging.getLogger('nlpy.algencan')
    algencanlogger.addHandler(hndlr)
    algencanlogger.propagate = False
    algencanlogger.setLevel(logging.INFO)
    if options.print_level >= 4:
        algencanlogger.setLevel(logging.DEBUG)

# Solve each problem in turn.
for ProblemName in args:
    nlp = amplpy.MFAmplModel(ProblemName)         # Create a model

    algencan = pass_to_algencan(nlp, **opts)

#     if multiple_problems:
#         problemName = os.path.splitext(os.path.basename(ProblemName))[0]
#         nlpylogger.info(fmt % (problemName, nlp.n, algencan.iter, algencan.nlp.Hprod,
#                                 algencan.f, repr(algencan.status), total_time))


#if not multiple_problems and not error:
    # Output final statistics
    # Not implemented yet due to the python wrapper which is not designed to 
    # provide us with the final values x, f, g, # of iter, ...

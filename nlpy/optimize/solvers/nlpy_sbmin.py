#!/usr/bin/env python

from nlpy.model import amplpy
from nlpy.optimize.tr.trustregion import TrustRegionFramework as TR
from nlpy.optimize.tr.trustregion import TrustRegionBQP as TRSolver
from nlpy.optimize.solvers.sbmin import SBMINFramework
from nlpy.tools.timing import cputime
import numpy
import sys

def pass_to_sbmin(nlp, showbanner=True):

    if nlp.m > 0:         # Check for constrained problem
        sys.stderr.write('%s has %d general constraints\n' % (ProblemName, nlp.m))
        return None

    t = cputime()
    tr = TR(eta1=0.25, eta2=0.75, gamma1=0.0625, gamma2=2)

    # we select a trust-region subproblem solver of our choice.
    SBMIN = SBMINFramework(nlp, tr, TRSolver, maxiter = 1000, verbose=False)
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

for ProblemName in sys.argv[1:]:
    nlp = amplpy.AmplModel(ProblemName)         # Create a model
    print nlp.Uvar
    print nlp.Lvar
    SBMIN = pass_to_sbmin(nlp, showbanner=True)
    #nlp.writesol(SBMIN.x, nlp.pi0, 'And the winner is')    # Output "solution"
    nlp.close()                                 # Close connection with model

# Plot the evolution of the trust-region radius on the last problem
if SBMIN is not None:
    try:
        import pylab
    except:
        sys.stderr.write('If you had pylab installed, you would be looking ')
        sys.stderr.write('at a plot of the evolution of the trust-region ')
        sys.stderr.write('radius, right now.\n')
        sys.exit(0)
    radii = numpy.array(SBMIN.radii, 'd')
    print radii
    #pylab.plot(numpy.where(radii < 100, radii, 100))
    pylab.plot(radii)
    pylab.title('Trust-region radius')
    pylab.show()

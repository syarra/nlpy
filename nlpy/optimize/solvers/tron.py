"""
 TRON
 M. P. Friedlander and D. Orban, Banff, June 2012
"""
from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.tools import norms
from nlpy.tools.timing import cputime
from nlpy.tools.exceptions import UserExitRequest
from nlpy.optimize.solvers.pytron import dtron
from math import sqrt
import numpy
import logging
import pdb
from nlpy.optimize.solvers.lbfgs import LBFGS
from nlpy.optimize.solvers.lsr1 import LSR1

__docformat__ = 'restructuredtext'

class troninter:

    def __init__(self, n):
        self.n       = n
        self.xc      = numpy.empty(n)
        self.s       = numpy.empty(n)
        self.indfree = numpy.empty(n)
        self.isave   = numpy.zeros(3, dtype=numpy.int32)
        self.dsave   = numpy.empty(3)
        self.wa      = numpy.empty(7*n)
        self.wx      = numpy.empty(n)
        self.wy      = numpy.empty(n)
        self.iwa     = numpy.empty(3*n, dtype=numpy.int32)
        self.cgiter  = 0

    def solve(self, task, x, xl, xu, f, g, aprod, delta,
              frtol=1.0e-12, fatol=0.0, fmin=-1.0e+32, cgtol=0.1, itermax=None):

        if itermax is None:
            itermax = self.n

        x, delta, task, self.xc, self.s, self.indfree, self.isave, self.dsave, \
            self.wa, self.wx, self.wy, self.iwa = \
            dtron(x, xl, xu, f, g, aprod, delta, task, self.xc, self.s,
                  self.indfree, self.isave, self.dsave, self.wa, self.wx,
                  self.wy, self.iwa, frtol, fatol, fmin, cgtol, itermax)

        task = task.strip()
        predred = self.dsave[2]
        cgiter = self.isave[2] # record the current no. of CG its
        return (task, x, predred, delta, cgiter)


class TronFramework:
    """
    An abstract framework Tron. Instantiate using

    `TRON = TronFramework(nlp)`

    :parameters:

        :nlp:   a :class:`NLPModel` object representing the problem. For
                instance, nlp may arise from an AMPL model

    :keywords:

        :x0:           starting point                    (default nlp.x0)
        :reltol:       relative stopping tolerance       (default `nlp.stop_d`)
        :abstol:       absolute stopping tolerance       (default 1.0e-6)
        :maxit:        maximum number of iterations      (default max(1000,10n))
        :inexact:      use inexact Newton stopping tol   (default False)
        :logger_name:  name of a logger object that can be used in the post
                       iteration                         (default None)
        :verbose:      print log if True                 (default True)

    Once a `TronFramework` object has been instantiated and the problem is
    set up, solve problem by issuing a call to `TRON.solve()`. The algorithm
    stops as soon as the Euclidian norm of the gradient falls below

        ``max(abstol, reltol * g0)``

    where ``g0`` is the Euclidian norm of the projected gradient at the
    initial point.
    """

    def __init__(self, nlp, **kwargs):

        self.nlp    = nlp
        self.iter   = 0         # Iteration counter
        self.total_cgiter = 0
        self.x      = kwargs.get('x0', self.nlp.x0.copy())
        self.maxit  = kwargs.get('maxit', max(1000, 10*self.nlp.n))
        self.f      = None
        self.f0     = None
        self.g      = None
        self.gpnorm = None
        self.task   = None

        self.tron = troninter(nlp.n)

        self.reltol  = kwargs.get('reltol', 1.0e-7)#self.nlp.stop_d)
        self.abstol  = kwargs.get('abstol', 1.0e-7)
        self.verbose = kwargs.get('verbose', True)
        self.logger  = kwargs.get('logger', None)

        self.format  = '%-5d  %9.2e  %7.1e  %5i  %8.1e  %8.1e'
        self.format0 = '%-5d  %9.2e  %7.1e  %5s  %8s  %8s'
        self.hformat = '%-5s  %9s  %7s  %5s  %8s  %8s'
        self.header  = self.hformat % ('Iter','f(x)','|Pg(x)|','cg','rho','Radius')
        self.hlen    = len(self.header)
        self.hline   = '-' * self.hlen

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlpy.tron')
        self.log = logging.getLogger(logger_name)
        self.log.addHandler(logging.NullHandler())
        if not self.verbose:
            self.log.propagate=False

    def _task_is(self, task):
        return self.task[0] == task[0]

    def _gpnorm2(self, x, g, Lvar, Uvar):
        """
        Compute 2-norm of the projected gradient.
        """
        ineq =  Lvar != Uvar
        lowr =  numpy.logical_and(x == Lvar, ineq)
        uppr =  numpy.logical_and(x == Uvar, ineq)
        free =  numpy.logical_and(numpy.logical_and(~lowr, ~uppr), ineq)
        gpnrm2  = norms.norm2( numpy.minimum( g[lowr], 0. ) )**2
        gpnrm2 += norms.norm2( numpy.maximum( g[uppr], 0. ) )**2
        gpnrm2 += norms.norm2(                g[free]       )**2
        return sqrt(gpnrm2)

    def hprod(self, v, **kwargs):
        """
        Default hprod based on nlp's hprod. User should overload to
        provide a custom routine, e.g., a quasi-Newton approximation.
        """
        return self.nlp.hprod(self.x, self.nlp.pi0, v)

    def precon(self, v, **kwargs):
        """
        Generic preconditioning method---must be overridden.
        Not yet implemented.
        """
        return v

    def PostIteration(self, **kwargs):
        """
        Override this method to perform work at the end of an iteration. For
        example, use this method for preconditioners that need updating,
        e.g., a limited-memory BFGS preconditioner.
        """
        return None

    def Solve(self, **kwargs):

        nlp = self.nlp

        # Project initial point into the box.
        self.x[self.x < nlp.Lvar] = nlp.Lvar[self.x < nlp.Lvar]
        self.x[self.x > nlp.Uvar] = nlp.Uvar[self.x > nlp.Uvar]
        self.x_old = self.x.copy()

        # Gather initial information.
        self.f       = self.nlp.obj(self.x)
        self.f0      = self.f
        self.g       = self.nlp.grad(self.x)
        self.gpnorm  = self._gpnorm2(self.x, self.g, nlp.Lvar, nlp.Uvar)
        self.gpnorm0 = self.gpnorm

        stoptol = max(self.abstol, self.reltol * self.gpnorm)
        exitUser = exitInner = False
        exitOptimal = self.gpnorm <= stoptol
        exitIter = self.iter >= self.maxit
        cgiter_old = 0

        t = cputime()

        # Print out header and initial log.
        if self.iter % 20 == 0 and self.verbose:
            self.log.info(self.hline)
            self.log.info(self.header)
            self.log.info(self.hline)
            self.log.info(self.format0 % (self.iter, self.f,
                                          self.gpnorm, '', '', ''))

        self.task = 'START'
        delta = norms.norm2(self.g)  # Initial trust-region radius.

        while not (exitUser or exitOptimal or exitIter or exitInner):

            #print self.task
            self.task, self.x, predred, delta, cgiter = \
                self.tron.solve(self.task, self.x, nlp.Lvar, nlp.Uvar, self.f, self.g,
                                self.hprod, delta, frtol=1e-12,#self.reltol,
                                fatol=1e-12,#self.abstol,
                                itermax=self.maxit)

            if self._task_is('F'):
                f_old = self.f
                self.f = self.nlp.obj(self.x)
            elif self._task_is('G'):
                self.g_old = self.g.copy()
                self.g = self.nlp.grad(self.x)
                self.gpnorm = self._gpnorm2(self.x, self.g, nlp.Lvar, nlp.Uvar)

            else:
                # Calculate the number of CG iterations made in this inner iteration.
                cgiter_periter = cgiter - cgiter_old
                cgiter_old = cgiter

                # Chechs for convergence
                self.iter += 1
                exitIter = self.iter >= self.maxit
                exitOptimal = self.gpnorm <= stoptol
                exitInner = self.task[:4] == 'CONV' # Difference between 2 successive iterates
                                                    # is sufficiently small.

                try:
                    self.PostIteration()
                except UserExitRequest:
                    exitUser = True

                # Keep current iterate to use in the next QN update
                self.x_old = self.x.copy()

                # Print out header, say, every 20 iterations
                if self.iter % 20 == 0 and self.verbose:
                    self.log.info(self.hline)
                    self.log.info(self.header)
                    self.log.info(self.hline)

                if self.verbose:
                    self.log.info(self.format % (self.iter, self.f,
                                                 self.gpnorm, cgiter_periter,
                                                 (f_old-self.f)/predred, delta))

        self.tsolve = cputime() - t    # Solve time 

        # Set final solver status.
        if exitUser:
            self.status = 'usr'
        elif exitOptimal:
            self.status = 'opt'
        elif exitInner:
            self.status = 'ropt'
        else: # exitIter
            self.status = 'itr'


class TronLqnFramework(TronFramework):
    """
    Class SBMINLqnFramework is a subclass of SBMINFramework. The method is
    based on a trust-region-based algorithm for nonlinear box constrained
    programming.
    The only difference is that a limited-memory Quasi-Newton Hessian
    approximation is used and maintained along the iterations. See class
    SBMINFramework for more information.
    """
    def __init__(self, nlp, **kwargs):

        qn = kwargs.get('quasi_newton','LBFGS')
        TronFramework.__init__(self, nlp, **kwargs)
        self.lqn = eval(qn+'(nlp.n, npairs=5, scaling=True)')


    def hprod(self, v, **kwargs):
        return self.lqn.matvec(v)


    def PostIteration(self, **kwargs):
        """
        This method updates the limited-memory quasi-Newton Hessian by appending
        the most recent (s,y) pair to it and possibly discarding the oldest one
        if all the memory has been used.
        """
        # Quasi-Newton approximation update
        s = self.x - self.x_old
        y = self.g - self.g_old
        self.lqn.store(s, y)



if __name__ == '__main__':
    import nlpy_tron


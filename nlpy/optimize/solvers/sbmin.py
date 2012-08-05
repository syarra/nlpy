# -*- coding: utf-8 -*-
"""
SBMIN
A Trust-Region Method for Bound-Constrained Optimization.
"""

from nlpy.optimize.solvers import lbfgs    # For Hessian Approximate
from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.tools.norms import norm_infty
from nlpy.tools.timing import cputime
from nlpy.tools.exceptions import UserExitRequest
import numpy as np
import logging
import pdb
from math import sqrt
from nlpy.model import NLPModel
from nlpy.krylov.linop import SimpleLinearOperator

__docformat__ = 'restructuredtext'

def FormEntireMatrix(on,om,Jop):
    J = np.zeros([om,on])
    for i in range(0,on):
        v = np.zeros(on)
        v[i] = 1.
        J[:,i] = Jop * v
    return J

class SBMINFramework(object):
    """
    An abstract framework for a trust-region-based algorithm for the nonlinear
    bound-constrained optimization problem

        minimize    f(x)
        subject to  l <= x <= u

    where some components of l and/or u may be infinite.

    Instantiate using

        `SBMIN = SBMINFramework(nlp, TR, TrSolver)`

    :parameters:

        :nlp:   a :class:`NLPModel` object representing the problem. For
                instance, nlp may arise from an AMPL model
        :TR:    a :class:`TrustRegionFramework` object
        :TrSolver:   a :class:`TrustRegionSolver` object.


    :keywords:

        :x0:           starting point                    (default nlp.x0)
        :reltol:       relative stopping tolerance       (default 1.0e-5)
        :maxiter:      maximum number of iterations      (default 10n)
        :logger_name:  name of a logger object that can be used in the post
                       iteration                         (default None)
        :verbose:      print some info if True                 (default True)

    Once a `SBMINFramework` object has been instantiated and the problem is
    set up, solve problem by issuing a call to `SBMIN.solve()`. The algorithm
    stops as soon as the infinity norm of the projected gradient into the
    feasible box falls below `reltol`

    """

    def __init__(self, nlp, TR, TrSolver, **kwargs):

        self.nlp    = nlp
        self.TR     = TR
        self.TrSolver = TrSolver
        self.solver   = None    # Will point to solver data in Solve()
        self.iter   = 0         # Iteration counter
        self.x0      = kwargs.get('x0', self.nlp.x0.copy())
        self.x = None
        self.f      = None
        self.f0     = kwargs.get('f0', None)
        self.g      = kwargs.get('g0', None)
        self.pgnorm  = None
        self.tsolve = 0.0
        self.true_step = None

        # Options for Nocedal-Yuan backtracking
        self.ny      = kwargs.get('ny', False)
        self.nbk     = kwargs.get('nbk', 5)
        self.alpha   = 1.0

        # Use magical steps to update slack variables
        self.magic_steps_cons = kwargs.get('magic_steps_cons',False)
        self.magic_steps_agg = kwargs.get('magic_steps_agg',False)

        # If both options are set, use the aggressive type
        if self.magic_steps_agg and self.magic_steps_cons:
            self.magic_steps_cons = False

        # Options for non monotone descent strategy
        self.monotone = kwargs.get('monotone', True)
        self.nIterNonMono = kwargs.get('nIterNonMono', 25)

        self.abstol  = kwargs.get('abstol', 1.0e-8)
        self.reltol  = kwargs.get('reltol', 1.0e-5)
        self.maxiter = kwargs.get('maxiter', 10*self.nlp.n)
        self.verbose = kwargs.get('verbose', True)
        self.total_bqpiter = 0

        self.hformat = '%-5s  %9s  %7s  %5s  %8s  %8s  %4s'
        self.header  = self.hformat % ('     Iter','f(x)','|g(x)|','bqp',
                                       'rho','Radius','Stat')
        self.hlen   = len(self.header)
        self.hline  = '     '+'-' * self.hlen
        self.format = '     %-5d  %9.2e  %7.1e  %5d  %8.1e  %8.1e  %4s'
        self.format0= '     %-5d  %9.2e  %7.1e  %5s  %8s  %8s  %4s'
        self.radii = None

        # Setup the logger. Install a NullHandler if no output needed.
        logger_name = kwargs.get('logger_name', 'nlpy.sbmin')
        self.log = logging.getLogger(logger_name)
        if not self.verbose:
            self.log.propagate=False


    def hprod(self, v, **kwargs):
        """
        Default hprod based on nlp's hprod. User should overload to
        provide a custom routine, e.g., a quasi-Newton approximation.
        """
        return self.nlp.hprod(self.x, self.nlp.pi0, v)

    def project(self, x):
        "Project x into the bounds."
        return np.maximum(np.minimum(x, self.nlp.Uvar), self.nlp.Lvar)

    def projected_gradient(self, x, g):
        """
        Compute the projected gradient of f(x) into the feasible box

                   l <= x <= u
        """
        #p = x - g
        #med = np.maximum(np.minimum(p,self.nlp.Uvar),self.nlp.Lvar)
        #q = x - med
        #return q
        return x - self.project(x - g)


    def PostIteration(self, **kwargs):
        """
        Override this method to perform work at the end of an iteration. For
        example, use this method for updating a LBFGS Hessian
        """
        return None


    def Solve(self, **kwargs):

        nlp = self.nlp

        # Gather initial information.
        self.x = self.project(self.x0)
        if self.f0 is None:
            self.f0 = nlp.obj(self.x)
        self.f = self.f0

        if self.g is None:
            self.g = nlp.grad(self.x)

        self.pgnorm = norm_infty(self.projected_gradient(self.x,self.g))
        self.pg0 = self.pgnorm

        # Reset initial trust-region radius.
        self.TR.Delta = np.maximum(0.1 * self.pgnorm,.2)
        self.radii = [ self.TR.Delta ]

        # Initialize non-monotonicity parameters.
        if not self.monotone:
            self.log.info('Using Non monotone descent strategy')
            fMin = fRef = fCan = self.f0
            l = 0
            sigRef = sigCan = 0

        stoptol = self.reltol * self.pg0 + self.abstol
        cgtol = 1.0
        step_status = None
        exitIter = exitUser = exitTR = False
        exitOptimal = self.pgnorm <= stoptol
        status = ''

        # Print out header and initial log.
        self.log.info(self.hline)
        self.log.info(self.header)
        self.log.info(self.hline)
        self.log.info(self.format0 % (self.iter, self.f,
                                             self.pgnorm, '', '',
                                             '', ''))

        t = cputime()

        while not (exitUser or exitOptimal or exitIter or exitTR):

            self.iter += 1

            # Iteratively minimize the quadratic model in the trust region
            #          m(d) = <g, d> + 1/2 <d, Hd>
            #     s.t.     ll <= d <= uu
            qp = TrustBQPModel(nlp, self.x, self.TR.Delta, g_k=self.g)

            cgtol = max(1.0e-6, min(0.5 * cgtol, sqrt(self.pgnorm)))

            self.solver = self.TrSolver(qp, qp.grad)
            self.solver.Solve(reltol=cgtol)

            step = self.solver.step
            self.true_step = self.solver.step.copy()
            stepnorm = self.solver.stepNorm
            bqpiter = self.solver.niter

            # Obtain model value at next candidate
            m = self.solver.m

            self.total_bqpiter += bqpiter
            x_trial = self.x + step
            f_trial = nlp.obj(x_trial)

            # Aggressive magical steps
            # (i.e. the magical steps can influence the trust region size)
            if self.magic_steps_agg:
                x_inter = x_trial.copy()
                f_inter = f_trial
                g_inter = nlp.grad(x_inter)
                m_step = nlp.magical_step(x_inter, g_inter)
                x_trial = x_inter + m_step
                self.true_step += m_step
                f_trial = nlp.obj(x_trial)
                if f_trial <= f_inter:
                    # Safety check for machine-precision errors in magical steps
                    m = m - (f_inter - f_trial)
            #     self.log.debug('pred = %20.16g, pred increase = %20.16g' % (self.solver.m, f_inter - f_trial))
            # else:
            #     self.log.debug('ared = %20.16f' % (self.f - f_trial))

            rho  = self.TR.Rho(self.f, f_trial, m)

            if not self.monotone:
                rhoHis = (fRef - f_trial)/(sigRef - m)
                rho = max(rho, rhoHis)

            step_status = 'Rej'

            if rho >= self.TR.eta1:

                # Trust-region step is successful
                self.TR.UpdateRadius(rho, stepnorm)
                self.x = x_trial
                self.f = f_trial
                self.g = nlp.grad(self.x)

                if self.magic_steps_cons:
                    m_step = nlp.magical_step(self.x, self.g)
                    self.x += m_step
                    self.true_step += m_step
                    self.f = nlp.obj(self.x)
                    self.g = nlp.grad(self.x)

                self.pgnorm = norm_infty(self.projected_gradient(self.x,self.g))
                step_status = 'Acc'

                # Update non-monotonicity parameters.
                if not self.monotone:
                    sigRef = sigRef - m
                    sigCan = sigCan - m
                    if f_trial < fMin:
                        fCan = f_trial
                        fMin = f_trial
                        sigCan = 0
                        l = 0
                    else:
                        l = l + 1

                    if f_trial > fCan:
                        fCan = f_trial
                        sigCan = 0

                    if l == self.nIterNonMono:
                        fRef = fCan
                        sigRef = sigCan

            else:

                # Attempt Nocedal-Yuan backtracking if requested
                if self.ny:
                    alpha = self.alpha
                    slope = np.dot(self.g, step)
                    bk = 0

                    while bk < self.nbk and \
                            f_trial >= self.f + 1.0e-4 * alpha * slope:
                        bk = bk + 1
                        alpha /= 1.5
                        x_trial = self.x + alpha * step
                        f_trial = nlp.obj(x_trial)

                    if f_trial >= self.f + 1.0e-4 * alpha * slope:
                        # Backtrack failed to produce an improvement,
                        # keep the current x, f, and g.
                        # (Monotone strategy)
                        step_status = 'N-Y Rej'
                    else:
                        # Backtrack succeeded, update the current point
                        self.true_step *= alpha
                        self.x = x_trial
                        self.f = f_trial
                        self.g = nlp.grad(self.x)

                        # Conservative magical steps can also apply if backtracking succeeds
                        if self.magic_steps_cons:
                            m_step = nlp.magical_step(self.x, self.g)
                            self.x += m_step
                            self.true_step += m_step
                            self.f = nlp.obj(self.x)
                            self.g = nlp.grad(self.x)

                        self.pgnorm = norm_infty(self.projected_gradient(self.x,self.g))
                        step_status = 'N-Y Acc'

                    # Update the TR radius regardless of backtracking success
                    self.TR.Delta = alpha * stepnorm

                else:
                    # Trust-region step is unsuccessful
                    self.TR.UpdateRadius(rho, stepnorm)

            self.step_status = step_status
            self.radii.append(self.TR.Delta)
            status = ''
            try:
                self.PostIteration()
            except UserExitRequest:
                status = 'usr'

            # Print out header, say, every 20 iterations
            if self.iter % 20 == 0:
                self.log.info(self.hline)
                self.log.info(self.header)
                self.log.info(self.hline)

            pstatus = step_status if step_status != 'Acc' else ''
            self.log.info(self.format % (self.iter, self.f,
                          self.pgnorm, bqpiter, rho,
                          self.radii[-2], pstatus))
            # if self.iter >= 100:
            #     self.log.debug('Detail f = %16.12f' % self.f)

            exitOptimal = self.pgnorm <= stoptol
            exitIter    = self.iter > self.maxiter
            exitTR      = self.TR.Delta <= 10.0 * self.TR.eps
            exitUser    = status == 'usr'

        self.tsolve = cputime() - t    # Solve time

        # Set final solver status.
        if exitUser:
            pass
        elif exitOptimal:
            status = 'opt'
        elif exitTR:
            status = 'tr'
        else: # self.iter > self.maxiter:
            status = 'itr'
        self.status = status


class TrustBQPModel(NLPModel):
    """
    Class for defining a Model to pass to BQP solver:
                min     m(xk + s) = g's + 1/2 s'Hs
                s.t.       l <= xk + s <= u
                           || s ||_∞  <= delta

    where `g` is the gradient evaluated at xk
    and `H` is  the Hessian evaluated at xk.
    """

    def __init__(self, nlp, xk, delta, **kwargs):

        Lvar = np.maximum(nlp.Lvar - xk, -delta)
        Uvar = np.minimum(nlp.Uvar - xk, delta)

        NLPModel.__init__(self, n=nlp.n, m=nlp.m, name='TrustRegionSubproblem',
                          Lvar=Lvar, Uvar=Uvar)
        self.nlp = nlp
        self.x0 = np.zeros(self.nlp.n)
        self.xk = xk.copy()
        self.delta = delta
        self.gk = kwargs.get('gk', None)
        if self.gk == None:
            self.gk = self.nlp.grad(self.xk)

    def obj(self, x, **kwargs):
        Hx = self.nlp.hprod(self.xk, None, x)
        qapprox = np.dot(self.gk, x)
        qapprox += .5 * np.dot(x, Hx)

        return qapprox

    def grad(self, x, **kwargs):
        g = self.gk.copy()
        g += self.nlp.hprod(self.xk, None, x)
        return g

    def hprod(self, x, pi, p, **kwargs):
        return self.nlp.hprod(self.xk, None, p)


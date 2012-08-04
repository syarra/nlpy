"""
This module implements a matrix-free active-set method for the
bound-constrained quadratic program

    minimize  g'x + 1/2 x'Hx  subject to l <= x <= u,

where l and u define a (possibly unbounded) box. The method
implemented is that of More and Toraldo described in

    J. J. More and G. Toraldo, On the solution of large
    quadratic programming problems with bound constraints,
    SIAM Journal on Optimization, 1(1), pp. 93-113, 1991.
"""

from nlpy.krylov.pcg   import TruncatedCG
from nlpy.krylov.linop import SimpleLinearOperator
from nlpy.krylov.linop import SymmetricallyReducedLinearOperator as ReducedHessian
from nlpy.tools.utils import identical, where, NullHandler
from nlpy.tools.exceptions import InfeasibleError, UserExitRequest
import numpy as np
import logging
import warnings

import ipdb

__docformat__ = 'restructuredtext'

def FormEntireMatrix(on,om,Jop):
    J = np.zeros([om,on])
    for i in range(0,on):
        v = np.zeros(on)
        v[i] = 1.
        J[:,i] = Jop * v
    return J


class SufficientDecreaseCG(TruncatedCG):
    """
    An implementation of the conjugate-gradient algorithm with
    a sufficient decrease stopping condition.

    :keywords:
        :cg_reltol: a relative stopping tolerance based on the decrease
                    of the quadratic objective function. The test is
                    triggered if, at iteration k,

                    q{k-1} - qk <= cg_reltol * max { q{j-1} - qj | j < k}

                    where qk is the value q(xk) of the quadratic objective
                    at the iterate xk.

    See the documentation of TruncatedCG for more information.

    Exit when either :
        - norm of gradient is less than eps_g times the initial gardient
          added in case whe have non stricly convex or non bounded problem
        - stalling process
          sufficient if the problem is convex and bounded
        - sufficient decrease
    """
    def __init__(self, g, H, **kwargs):
        TruncatedCG.__init__(self, g, H, **kwargs)
        self.name = 'Suff-CG'
        self.qOld = 0.0   # Initial value of quadratic objective.
        self.best_decrease = 0
        self.decrease_old = 0
        self.cg_reltol = kwargs.get('cg_reltol', 0.1)
        self.detect_stalling = kwargs.get('detect_stalling', True)
        self.Lvar = kwargs.get('Lvar', None)
        self.Uvar = kwargs.get('Uvar', None)
        self.x = kwargs.get('x', None)
        self.g0 = np.linalg.norm(g)

    def post_iteration(self):
        """
        Implement the sufficient decrease stopping condition. This test
        costs one dot product, five products between scalars and two
        additions of scalars.
        """
        p = self.p ; g = self.g ; pHp = self.pHp ; alpha = self.alpha
        s = self.step
        qCur = self.qval
        decrease = self.qOld - qCur
        Dqnorm = np.linalg.norm(g+self.H*(self.x+s))
        self.log.debug('qCur : %g' % qCur)

        projected_xps = np.minimum(self.Uvar, np.maximum(self.Lvar, self.x+s)) 

        if not identical(self.x+s,projected_xps):
            self.log.debug('CG stops with a constraint violation')
            raise UserExitRequest

        if Dqnorm <= 1e-7*self.g0:
            self.log.debug('CG stops with a sufficient decrease in the gradient norm')
            raise UserExitRequest

        if self.qval <= -1e+25:
            raise UserExitRequest

        #print 'old decrease:', self.decrease_old
        if self.iter != 1:
            if decrease >= 100*self.decrease_old:
                self.log.debug('CG stops with a sufficient decrease in the objective value')
                raise UserExitRequest

        self.decrease_old = decrease

        if self.detect_stalling:
            if decrease <= self.cg_reltol * self.best_decrease:
                self.log.debug('CG stops with a stalling status')
                raise UserExitRequest
            else:
                self.best_decrease = max(self.best_decrease, decrease)
        return None


class BQP(object):
    """
    A matrix-free active-set method for the bound-constrained quadratic
    program. May be use to solve trust-region subproblems in l-infinity
    norm.
    """
    def __init__(self, qp, **kwargs):
        super(BQP, self).__init__()
        if qp.m != 0:
            warnings.warn(('\nYou\'re trying to solve a constrained problem '
                           'with an unconstrained solver !\n'))

        self.qp = qp
        self.Lvar = qp.Lvar
        self.Uvar = qp.Uvar
        self.H = SimpleLinearOperator(qp.n, qp.n,
                                      lambda u: self.qp.hprod(self.qp.x0,
                                                              None,
                                                              u),
                                      symmetric=True)

        # Relative stopping tolerance in projected gradient iterations.
        self.pgrad_reltol = 0.25

        # Relative stopping tolerance in conjugate gradient iterations.
        self.cg_reltol = 0.1

        # Armijo-style linesearch parameter.
        self.armijo_factor = 1.0e-4

        self.verbose = kwargs.get('verbose',True)
        self.hformat = '          %-5s  %8s  %8s  %5s'
        self.header  = self.hformat % ('Iter','q(x)','|pg(x)|','cg')
        self.hlen   = len(self.header)
        self.hline  = '          '+'-' * self.hlen
        self.format = '          %-5d  %8.2e  %8.2e  %5d'
        self.format0= '          %-5d  %8.2e  %8.2e  %5s'

        # Create a logger for solver.
        self.log = logging.getLogger('nlpy.bqp')
        try:
            self.log.addHandler(logging.NullHandler()) # For Python 2.7.x
        except:
            self.log.addHandler(NullHandler()) # For Python 2.6.x (and older?)


    def check_feasible(self, x):
        """
        Safety function. Check that x is feasible with respect to the
        bound constraints.
        """
        Px = self.project(x)
        if not identical(x,Px):
            print 'Received infeasible point.'
            raise InfeasibleError
        return None


    def pgrad(self, x, g=None, active_set=None, check_feasible=True):
        """
        Compute the projected gradient of the quadratic at x.
        If the actual gradient is known, it should be passed using the
        `g` keyword.
        If the active set at x0 is known, it should be passed using the
        `active_set` keyword.
        Optionally, check that x is feasible.

        The projected gradient pg is defined componentwise as follows:

        pg[i] = min(g[i],0)  if x[i] is at its lower bound,
        pg[i] = max(g[i],0)  if x[i] is at its upper bound,
        pg[i] = g[i]         otherwise.
        """
        if check_feasible: self.check_feasible(x)

        if g is None: g = self.qp.grad(x)

        if active_set is None:
            active_set = self.get_active_set(x)
        lower, upper = active_set

        pg = g.copy()
        pg[lower] = np.minimum(g[lower],0)
        pg[upper] = np.maximum(g[upper],0)
        return pg


    def project(self, x):
        "Project a given x into the bounds in Euclidian norm."
        return np.minimum(self.qp.Uvar,
                          np.maximum(self.qp.Lvar, x))


    def get_active_set(self, x, check_feasible=True):
        """
        Return the set of active constraints at x.
        Optionally, check that x is feasible.

        Returns the couple (lower,upper) containing the indices of variables
        that are at their lower and upper bound, respectively.
        """
        lower_active = where(x==self.Lvar)
        upper_active = where(x==self.Uvar)
        return(lower_active, upper_active)


    def projected_linesearch(self, x, g, d, qval, step=1.0):
        """
        Perform an Armijo-like projected linesearch in the direction d.
        Here, x is the current iterate, g is the gradient at x,
        d is the search direction, qval is q(x) and
        step is the initial steplength.
        """

        qp = self.qp
        finished = False

        s = self.project(x + step * d) - x
        qTrial = qp.obj(x+s)
        slope = np.dot(g, s) 

        projected_xps = np.minimum(self.Uvar, np.maximum(self.Lvar, x+s)) 
        if not identical(x+s,projected_xps):
            interp = True
        else:
            interp = (qTrial - qval >= self.armijo_factor * slope)

        self.log.debug('qTrial0: %6.3e' % qTrial)
        if interp:
            # Perform projected Armijo linesearch in order to reduce the step
            # until a successful step is found
            search = True
            while search and step >= 1e-30:
                step /= 10
                s = self.project(x + step * d) - x
                projected_xps = np.minimum(self.Uvar, np.maximum(self.Lvar, x+s))
                if identical(x+s,projected_xps):
                    qTrial = qp.obj(x+s)
                    self.log.debug('qTrial interp: %6.3e' % qTrial)
                    slope = np.dot(g, s)
                    search = (qTrial - qval >= self.armijo_factor * slope)
            xTrial = x+s
            self.log.debug('\n')
        else:
            search = True
            steps = step
            while search and step <= 1e+30:
                step *= 10
                s = self.project(x + step * d) - x
                projected_xps = np.minimum(self.Uvar, np.maximum(self.Lvar, x+s))
                if identical(x+s,projected_xps):
                    qTrial = qp.obj(x+s)
                    self.log.debug('qTrial extrap: %6.3e'% qTrial)
                    slope = np.dot(g, s)
                    if qTrial - qval <= self.armijo_factor * slope:
                        search = True
                        steps = step
                else:
                    search = False
            step = steps
            xTrial = self.project(x + step * d)
            qTrial = qp.obj(xTrial)
            self.log.debug('qTrial extrap best: %6.3e \n' % qTrial)
        return (xTrial,qTrial)


    def projected_gradient(self, x0, g=None, active_set=None, qval=None, **kwargs):
        """
        Perform a sequence of projected gradient steps starting from x0.
        If the actual gradient at x is known, it should be passed using the
        `g` keyword.
        If the active set at x0 is known, it should be passed using the
        `active_set` keyword.
        If the value of the quadratic objective at x0 is known, it should
        be passed using the `qval` keyword.

        Return (x,(lower,upper)) where x is an updated iterate that satisfies
        a sufficient decrease condition or at which the active set, given by
        (lower,upper), settled down.
        """
        maxiter = kwargs.get('maxiter', 10)
        qp = self.qp

        if g is None:
            g = self.qp.grad(x0)

        if qval is None:
            qval = self.qp.obj(x0)

        if active_set is None:
            active_set = self.get_active_set(x0)
        lower, upper = active_set

        x = x0.copy()
        settled_down = False
        sufficient_decrease = False
        best_decrease = 0
        iter = 0

        while not settled_down and not sufficient_decrease and \
              iter < maxiter:

            iter += 1
            qOld = qval
            # TODO: Use appropriate initial steplength.
            (x, qval) = self.projected_linesearch(x, g, -g, qval)

            # Check decrease in objective.
            decrease = qOld - qval

            if decrease <= self.pgrad_reltol * best_decrease:
                sufficient_decrease = True
            best_decrease = max(best_decrease, decrease)

            # Check active set at updated iterate.
            lowerTrial, upperTrial = self.get_active_set(x)
            if identical(lower,lowerTrial) and identical(upper,upperTrial):
                settled_down = True
            lower, upper = lowerTrial, upperTrial
            #g = self.qp.grad(x)

        return (x, (lower, upper))


    def to_boundary(self, x, d, free_vars):
        """
        Given vectors `x` and `d` and some bounds on x,
        return a positive alpha such that

          `x + alpha * d = boundary
        """
        nonzeroind = d != 0.
        nonzerod = d[nonzeroind]

        if np.all(d[free_vars]==0.) == 0:

            # Follow the direction of negative curvature until it hits a bound
            aup = (self.Uvar[nonzeroind]-x[nonzeroind])/nonzerod
            aupp = aup[aup>-1e-60]
            alow = (self.Lvar[nonzeroind]-x[nonzeroind])/nonzerod
            alowp = alow[alow>-1e-60]
            if aupp.size != 0:
                aupmin = np.min(aupp)
                if alowp.size != 0:
                    alowmin = np.min(alowp)
                    alpha = np.minimum(alowmin, aupmin)
                else: alpha=aupmin
            else:
                if alowp.size != 0:
                    alpha = np.min(alowp)
                else: alpha=0.

            try:
                x += alpha*d
            except:
                pdb.set_trace()

        # Do another projected gradient update
        (x, (lower,upper)) = self.projected_gradient(x)

        return (x, (lower,upper))


    def solve(self, **kwargs):

        # Shortcuts for convenience.
        qp = self.qp
        n = qp.n
        maxiter = kwargs.get('maxiter', 5*n)
        self.stoptol = kwargs.get('stoptol', 1.0e-3)

        # Compute initial data.

        self.log.debug('q before initial x projection = %8.2g' % qp.obj(qp.x0))
        x = self.project(qp.x0)
        self.log.debug('q after initial x projection = %8.2g' % qp.obj(x))
        lower, upper = self.get_active_set(x)
        iter = 0

        # Compute stopping tolerance.
        g = qp.grad(x)
        gNorm = np.linalg.norm(g)
        pg = self.pgrad(x, g=g, active_set=(lower,upper))
        pgNorm = np.linalg.norm(pg)

        stoptol = self.stoptol*pgNorm #+ 1e-5
        self.log.debug('Main loop with iter=%d and pgNorm=%g' % (iter, pgNorm))

        exitOptimal = exitIter = False

        # Print out header and initial log.
        self.log.info(self.hline)
        self.log.info(self.header)
        self.log.info(self.hline)
        self.log.info(self.format0 % (iter,0.0,
                                             pgNorm, ''))

        while not (exitOptimal or exitIter):

            iter += 1
            if iter >= maxiter:
                exitIter = True
                continue

            # Projected-gradient phase: determine next working set.
            (x, (lower,upper)) = self.projected_gradient(x, g=g, active_set=(lower,upper))
            g = qp.grad(x)
            qval = qp.obj(x)
            self.log.debug('q after projected gradient = %8.2g' % qval)
            pg = self.pgrad(x, g=g, active_set=(lower,upper))
            pgNorm = np.linalg.norm(pg)

            if pgNorm <= stoptol:
                exitOptimal = True
                self.log.info(self.format % (iter, qval,
                              pgNorm, 0))

                continue

            # Conjugate gradient phase: explore current face.

            # 1. Obtain indices of the free variables.
            fixed_vars = np.concatenate((lower,upper))
            free_vars = np.setdiff1d(np.arange(n, dtype=np.int), fixed_vars)

            # 2. Construct reduced QP.
            self.log.debug('Starting CG on current face.')

            ZHZ = ReducedHessian(self.H, free_vars)
            Zg  = g[free_vars]
            cg = SufficientDecreaseCG(Zg, ZHZ, x=x[free_vars], Lvar=qp.Lvar[free_vars], Uvar=qp.Uvar[free_vars], detect_stalling=True)
            try:
                cg.Solve()
            except UserExitRequest:
                # CG is no longer making substantial progress.
                self.log.debug('CG is no longer making substantial progress (%d its)' % cg.niter)
                pass

            # At this point, CG returned from a clean user exit or
            # because its original stopping test was triggered.
            self.log.debug('CG stops after %d its with status=%s.' % (cg.niter,cg.status))

            # 3. Expand search direction.
            d = np.zeros(n)
            d[free_vars] = cg.step

            if cg.infDescent and cg.step.size != 0 and cg.dir.size !=0:
                self.log.debug('iter :%d  Negative curvature detected (%d its)' % (iter,cg.niter))

                nc_dir = np.zeros(n)
                nc_dir[free_vars] = cg.dir
                (x, (lower,upper)) = self.to_boundary(x,nc_dir,free_vars)
            else:
                # 4. Update x using projected linesearch with initial step=1.
                (x, qval) = self.projected_linesearch(x, g, d, qval)

            self.log.debug('q after first CG pass = %8.2g' % qp.obj(x))

            g = qp.grad(x)
            pg = self.pgrad(x, g=g, active_set=(lower,upper))
            pgNorm = np.linalg.norm(pg)

            if pgNorm <= stoptol:
                exitOptimal = True
                self.log.info(self.format % (iter, qval,
                              pgNorm, cg.niter))
                continue

            # Compare active set to binding set.
            lower, upper = self.get_active_set(x)

            if np.all(g[lower] >= 0) and np.all(g[upper] <= 0):
                # The active set agrees with the binding set.
                # Continue CG iterations with tighter tolerance.
                # This currently incurs a little bit of extra work
                # by instantiating a new CG object.
                self.log.debug('Active set and binding set match. Continuing CG.')

                fixed_vars   = np.concatenate((lower,upper))
                free_vars = np.setdiff1d(np.arange(n, dtype=np.int), fixed_vars)
                ZHZ = ReducedHessian(self.H, free_vars)
                Zg  = g[free_vars]
                s0 = cg.step[:]
                cg = SufficientDecreaseCG(Zg, ZHZ,  x=x[free_vars], Lvar=qp.Lvar[free_vars], Uvar=qp.Uvar[free_vars], detect_stalling=False, second_round=True)
                cg.Solve()

                self.log.debug('CG stops after %d its with status=%s.' % (cg.niter,cg.status))
                d = np.zeros(n)
                d[free_vars] = cg.step

                if cg.infDescent and cg.step.size != 0:
                    self.log.debug('iter :%d  Negative curvature detected (%d its)' % (iter,cg.niter))

                    nc_dir = np.zeros(n)
                    nc_dir[free_vars] = cg.dir
                    (x, (lower,upper)) = self.to_boundary(x,nc_dir,free_vars)
                    qval = qp.obj(x)
                else:
                    # 4. Update x using projected linesearch with initial step=1.
                    (x, qval) = self.projected_linesearch(x, g, d, qval)

                self.log.debug('q after second CG pass = %8.2g' % qp.obj(x))

                g = qp.grad(x)
                pg = self.pgrad(x, g=g, active_set=(lower,upper))
                pgNorm = np.linalg.norm(pg)

            # Exit if second CG pass results in optimality
            if pgNorm <= stoptol:
                exitOptimal = True

            self.log.info(self.format % (iter, qval,
                          pgNorm, cg.niter))

        self.exitOptimal = exitOptimal
        self.exitIter = exitIter
        self.niter = iter
        self.x = x
        self.qval = qval
        self.lower = lower
        self.upper = upper
        return



if __name__ == '__main__':
    import sys
    from nlpy.model import AmplModel

    qp = AmplModel(sys.argv[1])
    bqp = BQP(qp)
    bqp.solve(maxiter=50, stoptol=1.0e-8)
    print 'optimal = ', repr(bqp.exitOptimal)
    print 'niter = ', bqp.niter
    print 'solution: ', bqp.x
    print 'objective value: ', bqp.qval
    print 'vars on lower bnd: ', bqp.lower
    print 'vars on upper bnd: ', bqp.upper


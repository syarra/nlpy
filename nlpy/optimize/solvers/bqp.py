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
<<<<<<< HEAD
from nlpy.tools.utils import identical, where
=======
from nlpy.tools import identical, where
>>>>>>> upstream/optpkg
from nlpy.tools.exceptions import InfeasibleError, UserExitRequest
import numpy as np
import logging

import pdb

__docformat__ = 'restructuredtext'


class SufficientDecreaseCG(TruncatedCG):
    """
    An implementation of the conjugate-gradient algorithm with
    a sufficient decrease stopping condition.

    :keywords:
        :cg_reltol: a relative stopping tolerance based on the decrease
                    of the quadratic objective function. The test is
                    triggered if, at iteration k,

                    q{k-1} - qk <= cg_reltol * min { q{j-1} - qj | j < k}

                    where qk is the value q(xk) of the quadratic objective
                    at the iterate xk.

    See the documentation of TruncatedCG for more information.
    """
    def __init__(self, g, H, **kwargs):
        TruncatedCG.__init__(self, g, H, **kwargs)
        self.name = 'Suff-CG'
        self.qval = 0.0   # Initial value of quadratic objective.
        self.best_decrease = 0
        self.cg_reltol = kwargs.get('cg_reltol', 0.1)
        self.detect_stalling = kwargs.get('detect_stalling', True)


    def post_iteration(self):
        """
        Implement the sufficient decrease stopping condition. This test
        costs one dot product, five products between scalars and two
        additions of scalars.
        """
        if not self.detect_stalling: return None
        p = self.p ; g = self.g ; pHp = self.pHp ; alpha = self.alpha
        qOld = self.qval
        qCur = qOld + alpha * np.dot(g,p) + 0.5 * alpha*alpha * pHp
        decrease = qOld - qCur
        if decrease <= self.cg_reltol * self.best_decrease:
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
        self.qp = qp
        self.Lvar = qp.Lvar
        self.Uvar = qp.Uvar
        self.H = SimpleLinearOperator(qp.n, qp.n,
                                      lambda u: self.qp.hprod(self.qp.x0,
                                                              self.qp.pi0,
                                                              u),
                                      symmetric=True)

        # Relative stopping tolerance in projected gradient iterations.
        self.pgrad_reltol = 0.25

        # Relative stopping tolerance in conjugate gradient iterations.
        self.cg_reltol = 0.1

        # Armijo-style linesearch parameter.
        self.armijo_factor = 1.0e-4

        self.optimal = False

        # Create a logger for solver.
        self.log = logging.getLogger('bqp.solver')
        self.log.addHandler(logging.NullHandler())


    def check_feasible(self, x):
        """
        Safety function. Check that x is feasible with respect to the
        bound constraints.
        """
        Px = self.project(x)
        if not identical(x,Px):
            raise InfeasibleError, 'Received infeasible point.'
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
        if check_feasible: self.check_feasible(x)

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
        # TODO: Does it help to replace this with Bertsekas' modified
        #       Armijo condition?

        qp = self.qp
        finished = False

        # Perform projected Armijo linesearch.
        while not finished:

            xTrial = self.project(x + step * d)
            qTrial = qp.obj(xTrial)
            slope = np.dot(g, xTrial-x)
            print '  step=', step, ', slope=', slope

            if qTrial <= qval + self.armijo_factor * slope:
                finished = True
            else:
                step /= 3

        return (xTrial, qTrial)


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

        print 'Entering projected_gradient'
        print '  qval=', qval, 'lower=', lower, ', upper=', upper

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

            print '  qval=', qval, 'lower=', lower, ', upper=', upper
            print '  settled=', repr(settled_down), ', decrease=', repr(sufficient_decrease)

        return (x, (lower, upper))


    def solve(self, **kwargs):

        # Shortcuts for convenience.
        qp = self.qp
        n = qp.n
        maxiter = kwargs.get('maxiter', 10*n)
        self.stoptol = kwargs.get('stoptol', 1.0e-5)

        # Compute initial data.
        x = self.project(qp.x0)
        lower, upper = self.get_active_set(x)
        iter = 0

        # Compute stopping tolerance.
        g = qp.grad(x)
        gNorm = np.linalg.norm(g)
        stoptol = self.stoptol * gNorm

        pg = self.pgrad(x, g=g, active_set=(lower,upper))
        pgNorm = np.linalg.norm(pg)
        print 'Main loop with iter=%d and pgNorm=%g' % (iter, pgNorm)

        exitOptimal = exitIter = False

        #while pgNorm > stoptol and iter < maxiter:
        while not (exitOptimal or exitIter):

            iter += 1

            if iter >= maxiter:
                exitIter = True
                continue

            # Projected-gradient phase: determine next working set.
            (x, (lower,upper)) = self.projected_gradient(x, g=g, active_set=(lower,upper))
            g = qp.grad(x)
            qval = qp.obj(x)
            pg = self.pgrad(x, g=g, active_set=(lower,upper))
            pgNorm = np.linalg.norm(pg)
            print 'Main loop with iter=%d and pgNorm=%g' % (iter, pgNorm)

            if pgNorm <= stoptol:
                exitOptimal = True
                continue

            # Conjugate gradient phase: explore current face.

            # 1. Obtain indices of the free variables.
            fixed_vars = np.concatenate((lower,upper))
            free_vars = np.setdiff1d(np.arange(n, dtype=np.int), fixed_vars)

            # 2. Construct reduced QP.
            print 'Starting CG on current face.'
            ZHZ = ReducedHessian(self.H, free_vars)
            Zg  = g[free_vars]
            print 'Free vars: ', free_vars
            print 'ZHZ.shape: ', ZHZ.shape
            print 'len(Zg)  : ', len(Zg)
            cg = SufficientDecreaseCG(Zg, ZHZ)
            try:
                cg.Solve()
            except UserExitRequest:
                # CG is no longer making substantial progress.
                print 'CG is no longer making substantial progress (%d its)' % cg.niter
                pass

            # Temporary check.
            if cg.infDescent: print '  Negative curvature detected (%d its)' % cg.niter

            # At this point, CG returned from a clean user exit or
            # because its original stopping test was triggered.
            print '  CG stops after %d its with status=%s.' % (cg.niter,cg.status)

            # 3. Expand search direction.
            d = np.zeros(n)
            d[free_vars] = cg.step

            # 4. Update x using projected linesearch with initial step=1.
            (x, qval) = self.projected_linesearch(x, g, d, qval)
            g = qp.grad(x)
            pg = self.pgrad(x, g=g, active_set=(lower,upper))
            pgNorm = np.linalg.norm(pg)

            if pgNorm <= stoptol:
                exitOptimal = True
                continue

            # Compare active set to binding set.
            lower, upper = self.get_active_set(x)
            if np.all(g[lower] >= 0) and np.all(g[upper] <= 0):
                # The active set agrees with the binding set.
                # Continue CG iterations with tighter tolerance.
                # This currently incurs a little bit of extra work
                # by instantiating a new CG object.
                print 'Active set and binding set match. Continuing CG.'
                s0 = cg.step[:]
                cg = SufficientDecreaseCG(Zg, ZHZ, detect_stalling=False)
                cg.Solve(s0=s0)
                if cg.infDescent: print '    Negative curvature detected (%d its)' % cg.niter
                print '    CG stops after %d its with status=%s.' % (cg.niter,cg.status)

                d = np.zeros(n)
                d[free_vars] = cg.step

                # 4. Update x using projected linesearch with initial step=1.
                (x, qval) = self.projected_linesearch(x, g, d, qval)
                g = qp.grad(x)
                pg = self.pgrad(x, g=g, active_set=(lower,upper))
                pgNorm = np.linalg.norm(pg)


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


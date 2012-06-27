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

import pdb

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
        if self.detect_stalling: 

            p = self.p ; g = self.g ; pHp = self.pHp ; alpha = self.alpha
            qOld = self.qval
            qCur = qOld + alpha * np.dot(g,p) + 0.5 * alpha*alpha * pHp
            decrease = qOld - qCur
            if decrease <= self.cg_reltol * self.best_decrease:
                raise UserExitRequest
            else:
                self.best_decrease = max(self.best_decrease, decrease)

        return None



class BoundedCG(TruncatedCG):
    """
    An implementation of the conjugate-gradient algorithm which terminates 
    if the step violates user-defined bounds.

    See the documentation of TruncatedCG for more information.
    """
    def __init__(self, g, H, **kwargs):
        TruncatedCG.__init__(self, g, H, **kwargs)
        self.name = 'Bound-CG'
        self.detect_bounds = kwargs.get('detect_bounds', False)
        self.nviol = kwargs.get('nviol',1)
        self.s_l = kwargs.get('s_l', None)
        self.s_u = kwargs.get('s_u', None)


    def post_iteration(self):
        """
        Implement the stopping condition for violated bounds.
        """
        if self.detect_bounds:

            s = self.step
            s_l = self.s_l
            s_u = self.s_u
            l_viol = where(s < s_l)
            u_viol = where(s_u < s)
            if len(l_viol) + len(u_viol) >= self.nviol:
                self.log.debug('Too many bound violations detected: exiting.')
                raise UserExitRequest

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
        # TODO: Does it help to replace this with Bertsekas' modified
        #       Armijo condition?

        qp = self.qp
        finished = False

        # Perform projected Armijo linesearch.
        while not finished and step >= 1e-21:

            xTrial = self.project(x + step * d)
            qTrial = qp.obj(xTrial)
            slope = np.dot(g, xTrial-x)

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


    def projected_gradient_fast(self, x0, g=None, active_set=None, qval=None, **kwargs):
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

        This function is based on the Cauchy point calculator in TRON: 
        if a sufficient decrease is found for the first point computed, the 
        algorithm increases the step length as long as the sufficient decrease 
        condition remains satisfied.
        """
        beta = kwargs.get('beta',2)
        qp = self.qp

        if g is None:
            g = self.qp.grad(x0)

        if qval is None:
            qval = self.qp.obj(x0)

        if active_set is None:
            active_set = self.get_active_set(x0)
        lower, upper = active_set

        x = x0.copy()
        q0 = qval
        alpha = 1.0
        sufficient_decrease = False        
        iter = 0

        while not sufficient_decrease and alpha > 1e-20 and alpha < 1e+20:

            iter += 1

            xTrial = self.project(x - alpha*g)
            qval = qp.obj(xTrial)
            slope = np.dot(g, xTrial - x)

            if qval <= q0 + self.armijo_factor*slope:
                sufficient_decrease = True

            self.log.debug('alpha = %g, qdiff = %g' % (alpha, q0 + self.armijo_factor*slope - qval))

            if iter == 1:
                if sufficient_decrease == True:
                    sufficient_decrease = False
                    alpha *= beta
                else:
                    alpha /= beta
            else:
                if sufficient_decrease == True and alpha > 1.0:
                    sufficient_decrease = False
                    alpha *= beta
                elif sufficient_decrease == False and alpha > 1.0:
                    # Solution is the previous point we tried last iteration
                    sufficient_decrease = True
                    alpha /= beta
                    xTrial = self.project(x - alpha*g)
                elif sufficient_decrease == False and alpha < 1.0:
                    alpha /= beta
                # end if
            # end if

        # end while

        lower, upper = self.get_active_set(xTrial)

        return (xTrial, (lower, upper))


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

        stoptol = self.stoptol*pgNorm + 1e-5
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

            cg = SufficientDecreaseCG(Zg, ZHZ)
            try:
                cg.Solve()
            except UserExitRequest:
                # CG is no longer making substantial progress.
                self.log.debug('CG is no longer making substantial progress (%d its)' % cg.niter)
                pass

            # At this point, CG returned from a clean user exit or
            # because its original stopping test was triggered.
            self.log.debug('CG stops after %d its with status=%s.' % (cg.niter,cg.status))
            #if cg.status == 'residual small':
            #    self.log.debug('CG residual = %g, pHp = %g' % (cg.ry**0.5,cg.pHp))

            # 3. Expand search direction.
            d = np.zeros(n)
            d[free_vars] = cg.step

            if cg.infDescent and cg.step.size != 0 and cg.dir.size !=0:
                self.log.debug('iter :%d  Negative curvature detected (%d its)' % (iter,cg.niter))

                # (x, (lower,upper)) = self.to_boundary(x,d,free_vars)
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
                s0 = cg.step[:]
                cg = SufficientDecreaseCG(Zg, ZHZ, detect_stalling=False)
                cg.Solve(s0=s0)

                self.log.debug('CG stops after %d its with status=%s.' % (cg.niter,cg.status))
                d = np.zeros(n)
                d[free_vars] = cg.step

                if cg.infDescent and cg.step.size != 0:
                    self.log.debug('iter :%d  Negative curvature detected (%d its)' % (iter,cg.niter))
                    # (x, (lower,upper)) = self.to_boundary(x,d,free_vars)
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



class BQP_new(BQP):
    """
    This is a new version of the BQP class that incorporates more of the 
    ideas of the TRON solver (Lin & More, 1999) into the solve function.
    """

    def __init__(self, qp, **kwargs):
        BQP.__init__(self, qp, **kwargs)


    def projected_linesearch(self, x0, g, d, qval, active_set=None, **kwargs):
        """
        Perform an Armijo-like projected linesearch in the direction d.
        Here, x is the current iterate, g is the gradient at x,
        d is the search direction, and qval is q(x). 

        This function is based on the Cauchy point calculator in TRON: 
        if a sufficient decrease is found for the first point computed, the 
        algorithm increases the step length as long as the sufficient decrease 
        condition remains satisfied.
        """
        alpha = kwargs.get('alpha',1.0)
        beta = kwargs.get('beta',2.0)
        backtrack_only = kwargs.get('backtrack_only',False)
        qp = self.qp

        x = x0.copy()
        q0 = qval
        sufficient_decrease = False        
        iter = 0

        while not sufficient_decrease and alpha > 1e-20 and alpha < 1e+20:

            iter += 1

            xTrial = self.project(x + alpha*d)
            qval = qp.obj(xTrial)
            slope = np.dot(g, xTrial - x)

            if qval <= q0 + self.armijo_factor*slope:
                sufficient_decrease = True

            self.log.debug('alpha = %g, qdiff = %g' % (alpha, q0 + self.armijo_factor*slope - qval))

            if iter == 1:
                if sufficient_decrease == True and not backtrack_only:
                    sufficient_decrease = False
                    alpha *= beta
                elif sufficient_decrease == False:
                    alpha /= beta
            else:
                if sufficient_decrease == True and alpha > 1.0:
                    sufficient_decrease = False
                    alpha *= beta
                elif sufficient_decrease == False and alpha > 1.0:
                    # Solution is the previous point we tried last iteration
                    sufficient_decrease = True
                    alpha /= beta
                    xTrial = self.project(x + alpha*d)
                    qval = qp.obj(xTrial)
                elif sufficient_decrease == False and alpha < 1.0:
                    alpha /= beta
                # end if
            # end if

        # end while

        lower, upper = self.get_active_set(xTrial)

        return (xTrial, qval, (lower, upper))


    def projected_gradient(self, x0, g=None, active_set=None, qval=None, **kwargs):
        """
        Perform a sequence of projected gradient steps.

        The process is repeated for a fixed number of iterations or until the 
        active set settles down and the decrease is well below the best.
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
            (x, qval, new_active_set) = self.projected_linesearch(x, g, -g, qval, active_set)

            # Check decrease in objective.
            decrease = qOld - qval

            if decrease <= self.pgrad_reltol * best_decrease:
                sufficient_decrease = True
            best_decrease = max(best_decrease, decrease)

            # Check active set at updated iterate.
            lowerTrial, upperTrial = new_active_set
            if identical(lower,lowerTrial) and identical(upper,upperTrial):
                settled_down = True
            lower, upper = lowerTrial, upperTrial
            #g = self.qp.grad(x)

        return (x, qval, (lower, upper))


    def solve(self, **kwargs):

        # Shortcuts for convenience.
        qp = self.qp
        n = qp.n
        maxiter = kwargs.get('maxiter', 5*n)
        self.stoptol = kwargs.get('stoptol', 1.0e-3)

        # Compute initial data.

        # self.log.debug('q before initial x projection = %8.2g' % qp.obj(qp.x0))
        x = self.project(qp.x0)
        qval = qp.obj(x)
        self.log.debug('q after initial x projection = %8.2g' % qval)
        lower, upper = self.get_active_set(x)
        iter = 0

        # Compute stopping tolerance.
        g = qp.grad(x)
        gNorm = np.linalg.norm(g)
        pg = self.pgrad(x, g=g, active_set=(lower,upper))
        pgNorm = np.linalg.norm(pg)

        stoptol = self.stoptol*pgNorm + 1e-5
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
            (x, qval, (lower,upper)) = self.projected_gradient(x, g=g, active_set=(lower,upper), qval=qval)
            g = qp.grad(x)
            # qval = qp.obj(x)
            max_step_l = self.Lvar - x
            max_step_u = self.Uvar - x
            self.log.debug('q after projected gradient = %8.2g' % qval)
            pg = self.pgrad(x, g=g, active_set=(lower,upper))
            pgNorm = np.linalg.norm(pg)

            if pgNorm <= stoptol:
                exitOptimal = True
                self.log.info(self.format % (iter, qval,
                              pgNorm, 0))

                continue

            # Conjugate gradient phase: explore current face and add more 
            # active constraints if necessary

            active_set_settled = False

            while not active_set_settled:

                # 1. Obtain indices of the free variables.
                fixed_vars = np.concatenate((lower,upper))
                free_vars = np.setdiff1d(np.arange(n, dtype=np.int), fixed_vars)

                # 2. Construct reduced QP.
                self.log.debug('Starting CG on current face.')

                ZHZ = ReducedHessian(self.H, free_vars)
                Zg  = g[free_vars]
                sl = max_step_l[free_vars]
                su = max_step_u[free_vars]

                cg = BoundedCG(Zg, ZHZ, detect_bounds=True, s_l=sl, s_u=su)
                try:
                    cg.Solve()
                except UserExitRequest:
                    # CG is no longer making substantial progress.
                    self.log.debug('CG is no longer making substantial progress (%d its)' % cg.niter)
                    pass

                # At this point, CG returned from a clean user exit or
                # because its original stopping test was triggered.
                self.log.debug('CG stops after %d its with status=%s.' % (cg.niter,cg.status))
                #if cg.status == 'residual small':
                #    self.log.debug('CG residual = %g, pHp = %g' % (cg.ry**0.5,cg.pHp))

                # 3. Expand search direction.
                d = np.zeros(n)
                d[free_vars] = cg.step

                if cg.infDescent and cg.step.size != 0 and cg.dir.size !=0:
                    self.log.debug('iter :%d  Negative curvature detected (%d its)' % (iter,cg.niter))

                    # (x, (lower,upper)) = self.to_boundary(x,d,free_vars)
                    nc_dir = np.zeros(n)
                    nc_dir[free_vars] = cg.dir
                    (x, (lowerTrial,upperTrial)) = self.to_boundary(x,nc_dir,free_vars)
                    # TODO: check if we can replace above step with another projected linesearch
                else:
                    # 4. Update x using projected linesearch with initial step=1.
                    (x, qval, (lowerTrial,upperTrial)) = self.projected_linesearch(x, g, d, qval, active_set=(lower,upper), backtrack_only=True)

                self.log.debug('q after CG pass = %8.2g' % qval)

                # Recompute gradient for next iteration
                g = qp.grad(x)
                pg = self.pgrad(x, g=g, active_set=(lower,upper))
                pgNorm = np.linalg.norm(pg)

                # Check if new active set matches old one
                if identical(lower,lowerTrial) and identical(upper,upperTrial):
                    active_set_settled = True
                lower, upper = lowerTrial, upperTrial

            # end while

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


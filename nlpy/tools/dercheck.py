"""
A simple derivative checker.
"""

import numpy as np
from numpy.linalg import norm
from math import sqrt
import logging

np.random.seed(0)
macheps = np.finfo(np.double).eps  # Machine epsilon.


class DerivativeChecker:

    def __init__(self, nlp, x, **kwargs):
        """
        The `DerivativeChecker` class provides facilities for verifying
        numerically the accuracy of first and second-order derivatives
        implemented in an optimization model.

        `nlp` should be a `NLPModel` and `x` is the point about which we are
        checking the derivative. See the documentation of the `check` method
        for available options.

        :keywords:
            :logger_name:  Name of a logger for output
        """
        # Tolerance for determining if gradient seems correct.
        self.tol  = kwargs.get('tol', 100*sqrt(macheps))

        # Finite difference interval. Scale by norm(x). Use the 1-norm
        # so as not to make small x "smaller".
        self.step = kwargs.get('step', sqrt(macheps))
        self.h = self.step * (1 + norm(x, 1))

        self.nlp = nlp
        self.x = x
        self.grad_errs = []
        self.jac_errs = []
        self.hess_errs = []
        self.chess_errs = []

        headfmt = '%4s  %4s        %22s  %22s  %7s'
        self.head = headfmt % ('Fun', 'Comp', 'Expected',
                               'Finite Diff', 'Rel.Err')
        self.d1fmt = '%4d  %4d        %22.15e  %22.15e  %7.1e'
        head2fmt = '%4s  %4s  %4s  %22s  %22s  %7s'
        self.head2 = head2fmt % ('Fun', 'Comp', 'Comp', 'Expected',
                                 'Finite Diff', 'Rel.Err')
        self.d2fmt = '%4d  %4d  %4d  %22.15e  %22.15e  %7.1e'
        head3fmt = '%17s %22s  %22s  %7s'
        self.head3 = head3fmt % ('Directional Deriv', 'Expected',
                                 'Finite Diff', 'Rel.Err')
        self.d3fmt = '%17s %22.15e  %22.15e  %7.1e'

        logger_name = kwargs.get('logger_name', 'nlpy.derchk')
        self.logger = logging.getLogger(logger_name)
        self.logger.addHandler(logging.NullHandler())

        return


    def check(self, **kwargs):
        """
        Perform derivative check. Recognized keyword arguments are:

        :keywords:
            :grad:      Check objective gradient  (default `True`)
            :hess:      Check objective Hessian   (default `True`)
            :jac:       Check constraints Hessian (default `True`)
            :chess:     Check constraints Hessian (default `True`)
            :store_all: Also store accurate ders  (default `False`)
        """

        grad = kwargs.get('grad', True)
        hess = kwargs.get('hess', True)
        jac  = kwargs.get('jac', True)
        chess = kwargs.get('chess', True)
        cheap = kwargs.get('cheap_check', False)
        self.store_all = kwargs.get('store_all', False)

        # Skip constraints if problem is unconstrained.
        jac = (jac and self.nlp.m > 0)
        chess = (chess and self.nlp.m > 0)

        if grad:
            if cheap:
                self.grad_errs = self.cheap_check_obj_gradient()
            else:
                self.grad_errs = self.check_obj_gradient()
        if jac:
            self.jac_errs = self.check_con_jacobian()
        if hess:
            self.hess_errs = self.check_obj_hessian()
        if chess:
            self.chess_errs = self.check_con_hessians()

        return


    def cheap_check_obj_gradient(self):
        nlp = self.nlp
        n = nlp.n
        fx = nlp.obj(self.x)
        gx = nlp.grad(self.x)
        h = self.h

        self.logger.debug('Objective gradient (cheap)')
        self.logger.debug(self.head3)

        dx  = np.random.standard_normal(n)
        dx /= norm(dx)
        xph = self.x.copy()
        xph += h*dx
        dfdx = (nlp.obj(xph) - fx)/h      # finite-difference estimate
        gtdx = np.dot(gx, dx)             # expected
        err  = max(abs(dfdx - gtdx)/(1 + abs(gtdx)),
                   abs(dfdx - gtdx)/(1 + abs(dfdx)))

        errs = (0, 0, gtdx, dfdx, err)

        if errs[-1] >= self.tol:
            self.logger.error(self.d1fmt % errs)
        else:
            self.logger.debug(self.d1fmt % errs)

        if self.store_all:
            return errs
        else:
            return []


    def check_obj_gradient(self):
        nlp = self.nlp
        n = nlp.n
        gx = nlp.grad(self.x)
        h = self.h
        errs = []

        self.logger.debug('Objective gradient')
        self.logger.debug(self.head3)

        # Check partial derivatives in turn.
        for i in xrange(n):
            xph = self.x.copy() ; xph[i] += h
            xmh = self.x.copy() ; xmh[i] -= h
            dfdxi = (nlp.obj(xph) - nlp.obj(xmh))/(2*h)
            err = abs(gx[i] - dfdxi)/max(1, abs(gx[i]))

            this_err = (0, i, gx[i], dfdxi, err)
            if self.store_all or err >= self.tol:
                errs.append(this_err)

            if err >= self.tol:
                self.logger.error(self.d1fmt % this_err)
            else:
                self.logger.debug(self.d1fmt % this_err)

        return errs


    def check_obj_hessian(self):
        nlp = self.nlp
        n = nlp.n
        Hx = nlp.hess(self.x)
        h = self.step
        errs = []

        self.logger.debug('Objective Hessian')
        self.logger.debug(self.head2)

        # Check second partial derivatives in turn.
        for i in xrange(n):
            xph = self.x.copy() ; xph[i] += h
            xmh = self.x.copy() ; xmh[i] -= h
            dgdx = (nlp.grad(xph) - nlp.grad(xmh))/(2*h)
            for j in range(i+1):
                dgjdxi = dgdx[j]
                err = abs(Hx[i, j] - dgjdxi)/max(1, abs(Hx[i, j]))

                this_err = (0, i, j, Hx[i, j], dgjdxi, err)
                if self.store_all or err >= self.tol:
                    errs.append(this_err)

                if err >= self.tol:
                    self.logger.error(self.d2fmt % this_err)
                else:
                    self.logger.debug(self.d2fmt % this_err)

        return errs


    def check_con_jacobian(self):
        nlp = self.nlp
        n = nlp.n ; m = nlp.m
        if m == 0: return []   # Problem is unconstrained.

        Jx = nlp.jac(self.x)
        h = self.step
        errs = []

        self.logger.debug('Constraints Jacobian')
        self.logger.debug(self.head)

        # Check partial derivatives of each constraint in turn.
        for i in xrange(n):
            xph = self.x.copy() ; xph[i] += h
            xmh = self.x.copy() ; xmh[i] -= h
            dcdx = (nlp.cons(xph) - nlp.cons(xmh))/(2*h)
            for j in range(m):
                dcjdxi = dcdx[j]
                err = abs(Jx[j, i] - dcjdxi) / max(1, abs(Jx[j, i]))

                this_err = (j+1, i, Jx[j, i], dcjdxi, err)
                if self.store_all or err >= self.tol:
                    errs.append(this_err)

                if err >= self.tol:
                    self.logger.error(self.d1fmt % this_err)
                else:
                    self.logger.debug(self.d1fmt % this_err)

        return errs


    def check_con_hessians(self):
        nlp = self.nlp
        n = nlp.n ; m = nlp.m
        h = self.step
        errs = []

        self.logger.debug('Constraints Hessians')
        self.logger.debug(self.head2)

        # Check each Hessian in turn.
        for k in range(m):
            y = np.zeros(m) ; y[k] = -1
            Hk = nlp.hess(self.x, y, obj_weight=0)

            # Check second partial derivatives in turn.
            for i in xrange(n):
                xph = self.x.copy() ; xph[i] += h
                xmh = self.x.copy() ; xmh[i] -= h
                dgdx = (nlp.igrad(k, xph) - nlp.igrad(k, xmh)) / (2 * h)
                for j in xrange(i + 1):
                    dgjdxi = dgdx[j]
                    err = abs(Hk[i, j] - dgjdxi) / max(1, abs(Hk[i, j]))

                    this_err = (k + 1, i, j, Hk[i, j], dgjdxi, err)
                    if self.store_all or err >= self.tol:
                        errs.append(this_err)

                    if err >= self.tol:
                        self.logger.error(self.d2fmt % this_err)
                    else:
                        self.logger.debug(self.d2fmt % this_err)

        return errs


if __name__ == '__main__':

    import sys
    from nlpy.model import AmplModel

    log = logging.getLogger('nlpy.derchk')
    log.setLevel(logging.ERROR)
    fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
    hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)
    log.addHandler(hndlr)

    nlp = AmplModel(sys.argv[1])
    print 'Checking at x = ', nlp.x0
    derchk = DerivativeChecker(nlp, nlp.x0)
    derchk.check()
    nlp.close()

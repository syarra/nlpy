# -*- coding: utf-8 -*-

from pykrylov.linop.blkop import BlockLinearOperator
from pykrylov.linop import PysparseLinearOperator
from pykrylov.linop import ZeroOperator, IdentityOperator, LinearOperator
from pykrylov.minres import Minres
from pykrylov.lls import LSMRFramework
from nlpy.model import AmplModel, MFAmplModel
from nlpy.tools.norms import norm2, norm_infty
from nlpy.tools.timing import cputime
try:
    from nlpy.linalg.pyma57 import PyMa57Context as LBLContext
except:
    from nlpy.linalg.pyma27 import PyMa27Context as LBLContext
from nlpy.optimize.solvers.lbfgs import LBFGS, InverseLBFGS
from nlpy.tools.exceptions import UserExitRequest
import pysparse.sparse.pysparseMatrix as ps

import numpy as np
import logging, sys

class RegSQPSolver(object):
    """
    A regularized SQP method for degenerate equality-constrained optimization.
    """

    def __init__(self, nlp, **kwargs):
        """
        Instantiate a regularized SQP framework for a given equality-constrained
        problem.

        :keywords:
            :nlp: `NLPModel` instance.
            :abstol: Absolute stopping tolerance
            :reltol: relative required accuracy for || [g-J'y ; c] ||
            :theta: sufficient decrease condition for the inner iterations
            :itermax: maximum number of iterations allowed
        """
        self.nlp = nlp
        self.x = nlp.x0.copy()
        print self.x
        self.y = nlp.pi0.copy()

        self.abstol = kwargs.get('abstol', 1.0e-6)
        self.reltol = kwargs.get('reltol', 1.0e-8)
        self.theta = kwargs.get('theta',0.99)
        self.itermax = kwargs.get('maxiter', max(100, 10*nlp.n))
        self.save_g = kwargs.get('save_g', False)

        # Grab logger if one was configured.
        logger_name = kwargs.get('logger_name', 'regsqp.solver')
        self.log = logging.getLogger(logger_name)
        self.verbose = kwargs.get('verbose', False)
        if not self.verbose:
            self.log.propagate = False

        # Set regularization parameters.
        self.rho = kwargs.get('rho', 1.0) ; self.rho_min = 1.0e-8
        self.delta = kwargs.get('delta', 1.0) ; self.delta_min = 1.0e-8

        # Check input parameters.
        if self.rho < 0.0: self.rho = 0.0
        if self.delta < 0.0: self.delta = 0.0

        # Initialize format strings for display
        fmt_hdr = '%-4s  %-9s ' + ' %-8s '*3
        self.header = fmt_hdr % ('Iter', 'Obj', 'Rho', 'Delta', 'cNorm')
        self.format  = '%-4d  %-9.2e ' + ' %-8.2e ' * 3
        return

    def cons(self, x):
        """
        Return the value of the equality constraints evaluated at x and
        reformulated so that the right-hand side if zero, i.e., the original
        constraints c(x) = c0 are reformulated as c(x) - c0 = 0.
        """
        return (self.nlp.cons(x) - self.nlp.Lcon)

    def jac(self, x, *args, **kwargs):
        return PysparseLinearOperator(self.nlp.jac(x))

    def hess(self, x, z=None, **kwargs):
        return self.nlp.hess(x,z,**kwargs)

    def initialize_coefficient_matrix(self):
        # [ H+ρI    J' ] [∆x] = [ -g + J'y ]
        # [    J     -δI ] [∆y]    [ -c       ]

        # Create some diagonal matrices for use in `update_linear_system`
        self.In = ps.PysparseIdentityMatrix(size=self.nlp.n)
        self.Im = ps.PysparseIdentityMatrix(size=self.nlp.m)
        K = ps.PysparseMatrix(nrow=self.nlp.n+self.nlp.m,ncol=self.nlp.n+self.nlp.m,symmetric=True)
        return K

    def update_linear_system(self, x, y, rho, delta, **kwargs):
        # [ H+ρI    J' ] [∆x] = [ -g + J'y ]
        # [    J     -δI ] [∆y]    [ -c       ]
        #
        # For now H is the exact Hessian of the Lagrangian
        # (not an approximation of it).
        self.log.debug('Entering Update Linear System')

        # Some shortcuts for convenience
        n = self.nlp.n; m = self.nlp.m
        H = self.hess(x, y) ; J = self.nlp.jac(x)
        self.K = ps.PysparseMatrix(nrow=n+m,ncol=n+m,symmetric=True)
        (val,irow,jcol) = H.find()
        self.K.put(val,irow.tolist(),jcol.tolist())
        self.K.addAtDiagonal(rho*np.ones(n))
        (val,irow,jcol) = J.find()
        self.K.put(val, (n+irow).tolist(), jcol.tolist() )
        self.K.put(-delta*np.ones(m),range(n,n+m),range(n,m+n))

        self.log.debug('Leaving Update Linear System')

        return

    def initialize_rhs(self):
        return np.zeros(self.nlp.n+self.nlp.m)

    def update_rhs(self, rhs, g, J, y, c):
        n = self.nlp.n
        rhs[:n] = -g + J.T*y
        rhs[n:] = -c
        return

    def update_rho_delta(self, rho, delta):
        rho_min = self.rho_min ; delta_min = self.delta_min

        if delta > 0:
            delta = delta/10
            delta = max(delta, delta_min)
        if rho > 0:
            rho = rho/10
            rho = max(rho, rho_min)

        return rho,delta

    def phi(self, x, y, rho, delta, x0):
        nlp = self.nlp
        c = self.cons(x)
        phi = nlp.obj(x) - np.dot(c,y) + 0.5*rho*norm2(x-x0)**2 + 0.5/delta*norm2(c)**2
        return phi

    def dphi(self, x, y, rho, delta, x0):
        nlp = self.nlp ; n = nlp.n ; m = nlp.m
        dphi = np.zeros(n+m)
        c = self.cons(x) ; J = self.jac(x)
        dphi[:n] = nlp.grad(x) - J.T*(y-c/delta) + rho*(x-x0)
        dphi[n:] = -c
        return dphi

    def backtracking_linesearch(self, x, y, dx, dy, rho, delta,
                                                 bkmax=5, armijo=1.0-4):
        """
        Perform a simple backtracking linesearch on the merit function
        from `x` along `step`.

        Return (new_x, new_y, phi, steplength), where `new_x = x + steplength * step`
        satisfies the Armijo condition and `phi` is the merit function value at
        this new point.
        """
        self.log.debug('Entering backtracking linesearch')

        xTrial = x + dx ; yTrial = y + dy
        phi = self.phi(x, y, rho, delta, x)
        phiTrial = self.phi(xTrial, yTrial, rho, delta, x)
        g = self.dphi(x, y, rho, delta, x)
        slope = np.dot(g, np.concatenate((dx,-dy)))

        bk = 0
        alpha = 1.0
        while bk < bkmax and \
                phiTrial >= phi + armijo * alpha * slope:
            bk = bk + 1
            alpha /= 1.2
            xTrial = x + alpha * dx ; yTrial = y + alpha * dy
            phiTrial = self.phi(xTrial, yTrial, rho, delta, x)

        self.log.debug('    alpha=%3.2e, phi0=%3.2e, phi=%3.2e'%(alpha,phi,phiTrial))
        self.log.debug('Leaving backtracking linesearch')

        return (xTrial, yTrial, phiTrial, alpha)

    def solveSystem(self, x, y, rho, delta, rhs, itref_threshold=1.0e-5, nitrefmax=5, **kwargs):

        self.log.debug('Solving linear system')

        # Perform sparsity pattern analysis on the system.
        self.LBL = LBLContext(self.K, factorize=True, sqd=True)
        if not self.LBL.isFullRank:
            factorized = False
            degenerate = False
            nb_bump = 0
            while not factorized and not degenerate:
                self.log.debug('    A correction of inertia is needed.')
                self.update_linear_system(x, y, rho, delta)
                self.LBL.factorize(self.K)
                factorized = True

                # If the augmented matrix does not have full rank, bump up the
                # regularization parameters.
                if not self.LBL.isFullRank:
                    if rho > 0:
                        rho *= 100
                    nb_bump += 1
                    degenerate = nb_bump > self.bump_max
                    factorized = False

        # Abandon if regularization is unsuccessful.
        if not self.LBL.isFullRank and degenerate:
            status = '    Unable to regularize sufficiently.'
            short_status = 'degn'
            finished = True
        else:
            self.LBL.solve(rhs)
            self.LBL.refine(rhs, tol=itref_threshold, nitref=nitrefmax)
            self.log.debug('    residual norm: %3.2e'%norm2(self.LBL.residual))
            status = None ; short_status = None
            finished = False

        (dx, dy) = self.get_dxy(self.LBL.x)

        return status, short_status, finished, dx, dy

    def test_solveSystem(self):
        rhs = np.sum(self.K.to_array(), axis=1)
        step = self.solveSystem(rhs, itref_threshold=1.0e-5, nitrefmax=5)
        print step
        return

    def solve(self, **kwargs):

        # Transfer pointers for convenience.
        itermax = self.itermax ; abstol = self.abstol
        rho = self.rho ; delta = self.delta
        nlp = self.nlp ; x = self.x ; y = self.y
        theta = self.theta

        # Get initial objective value
        self.f0 = nlp.obj(x)
        self.x_old = x.copy()
        self.g_old = nlp.grad(x)

        # Initialize right-hand side and coefficient matrix
        # of linear systems
        rhs = self.initialize_rhs()

        g = nlp.grad(x)
        J = self.jac(x)
        c = self.cons(x)
        cnorm0 = norm_infty(c)
        optimal = ( max(norm_infty(g-J.T*y),cnorm0)<= abstol )

        finished = False
        iter = 0
        setup_time = cputime()
        finished = False or optimal

        # Display initial header every so often.
        self.log.info(self.header)
        self.log.info('-' * len(self.header))
        self.log.info(self.format%(iter, self.f0,self.rho, self.delta,cnorm0))

        # Main loop.
        while not finished:

            # Step 2
            rho, delta = self.update_rho_delta(rho, delta)

            self.update_linear_system(x, y, rho, delta, **kwargs)
            self.update_rhs(rhs, g, J, y, c)

            status, short_status, finished, dx, dy = self.solveSystem(x,y,rho,delta,rhs)
            if finished==True:
                continue

            # Step 3
            epsilon = 10*delta # a better way to set epsilon dynamically ?

            # Step 4: Inner Iterations
            xTrial = x + dx ; yTrial = y + dy
            gTrial = nlp.grad(xTrial) ; JTrial = self.jac(xTrial)
            cTrial = self.cons(xTrial)
            Fnorm = max(norm_infty(g-J.T*y),norm_infty(c))
            FTrialnorm = max(norm_infty(gTrial-JTrial.T*yTrial),norm_infty(cTrial))

            inner_iter = 0
            while FTrialnorm > theta*Fnorm + epsilon and inner_iter < 20:
                self.log.debug('    Entering inner iterations loop')

                # Step 3: Compute a new direction p_j
                self.update_linear_system(xTrial, yTrial, rho, delta, **kwargs)
                self.update_rhs(rhs, gTrial, JTrial, yTrial, cTrial)

                status, short_status, finished, dx_trial, dy_trial = self.solveSystem(xTrial, yTrial, rho, delta, rhs)

                # Break inner iteration loop if inertia correction fails
                if finished==True:
                    break

                # Step 4: Backtracking a la Armijo
                (xTrial, yTrial, phi_kj, alpha) = self.backtracking_linesearch(xTrial, yTrial, dx_trial, dy_trial,
                                                                         rho, delta)

                #xTrial = x_kj.copy() ; yTrial = y_kj.copy()
                gTrial = nlp.grad(xTrial) ; JTrial = self.jac(xTrial)
                cTrial = self.cons(xTrial)
                FTrialnorm = max(norm_infty(gTrial-JTrial.T*yTrial),norm_infty(cTrial))
                inner_iter+=1

                if FTrialnorm <= theta*Fnorm + epsilon or inner_iter >= 20:
                        self.log.debug('    Leaving inner iterations loop')
                else:
                    try:
                        self.PostInnerIteration(xTrial, gTrial)
                    except UserExitRequest:
                        self.status = -3

            if finished==True:
                continue

            # Update values of the new iterate and compute stopping criterion.
            x = xTrial.copy()
            print 'x: ', x
            y = yTrial.copy()
            g = nlp.grad(x)
            J = self.jac(x)
            c = self.cons(x)
            cnorm = norm_infty(c)
            optimal = ( max(norm_infty(g-J.T*y),cnorm)<= abstol )

            try:
                self.PostIteration(x, g)
            except UserExitRequest:
                self.status = -3

            # Display initial header every so often.
            if iter % 50 == 49:
                self.log.info(self.header)
                self.log.info('-' * len(self.header))

            iter += 1
            self.log.info(self.format%(iter, nlp.obj(x), rho, delta,cnorm))

            if  optimal:
                status = 'Optimal solution found'
                short_status = 'opt'
                finished = True
                continue

            if iter >= itermax:
                status = 'Maximum number of iterations reached'
                short_status = 'iter'
                finished = True
                continue

        solve_time = cputime() - setup_time

        # Transfer final values to class members.
        self.x = xTrial.copy()
        self.y = yTrial.copy()
        self.f = nlp.obj(self.x)
        self.cnorm = cnorm
        self.optimal = optimal
        self.rho = rho
        self.delta = delta
        self.iter = iter
        self.solve_time = solve_time
        self.status = status
        self.short_status = short_status
        return

    def get_dxy(self, step):
        """
        Split `step` into steps along x and y.
        Outputs are *references*, not copies.
        """
        self.log.debug('Recovering step')
        n = self.nlp.n
        dx = step[:n]
        dy = -step[n:]
        return (dx, dy)

    def PostIteration(self, x, g, **kwargs):
        """
        Override this method to perform additional work at the end of a
        major iteration. For example, use this method to restart an
        approximate Hessian.
        """
        return None

    def PostInnerIteration(self, x, g, **kwargs):
        """
        Override this method to perform additional work at the end of a
        minor iteration. For example, use this method to restart an
        approximate Hessian.
        """
        return None


class RegSQPBFGSIterativeSolver(RegSQPSolver):
    """
    A regularized SQP method for degenerate equality-constrained optimization.
    Using an iterative method to solve the system.
    """

    def __init__(self, nlp, IterativeSolver, **kwargs):
        """
        Instantiate a regularized SQP iterative framework for a given equality-constrained
        problem.

        :keywords:
            :nlp: `NLPModel` instance.
            :abstol: Absolute stopping tolerance
            :reltol: relative required accuracy for || [g-J'y ; c] ||
            :theta: sufficient decrease condition for the inner iterations
            :itermax: maximum number of iterations allowed
            :maxit_refine: maximum number of iterative refinements per solve
            :iterative_solver: solver used to find steps
            :maxit_refine: maximum number of iterative refinements per solve
        """

        super(RegSQPBFGSIterativeSolver, self).__init__(nlp, **kwargs)

        # Create some DiagonalOperators and save them.
        self.Im = IdentityOperator(nlp.m)

        # System solve method.
        self.iterative_solver = IterativeSolver

        #
        self.Hessapp = LBFGS(nlp.n, npairs=kwargs.get('qn_pairs',5), scaling=True, **kwargs)
        self.HessInvApp = InverseLBFGS(nlp.n, npairs=kwargs.get('qn_pairs',5), scaling=True, **kwargs)

        return

    def jac(self, x, *args, **kwargs):
        return self.nlp.jac(x)

    def hprod(self, x, z, v, **kwargs):
        """
        Compute the Hessian-vector product of the Hessian of the augmented
        Lagrangian with arbitrary vector v.
        """
        w = self.Hessapp.matvec(v)
        return w

    def hreset(self):
        self.Hessapp.restart()
        return

    def hess(self, x, z=None, **kwargs):
        return LinearOperator(self.nlp.n, self.nlp.n, symmetric=True,
                         matvec=lambda u: self.hprod(x,z,u,**kwargs))

    def HinvOp(self, **kwargs):
        return LinearOperator(self.nlp.n, self.nlp.n, symmetric=True,
                         matvec=lambda u: self.HessInvApp(u,**kwargs))

    def update_rho_delta(self, rho, delta):
        delta_min = self.delta_min

        if delta > 0:
            delta = delta/10
            delta = max(delta, delta_min)

        return 0,delta

    def update_linear_system(self, x, y, rho, delta, **kwargs):
        # [ H+ρI    J' ] [∆x] = [ -g + J'y ]
        # [    J     -δI ] [∆y]    [ -c       ]
        #
        # For now H is the exact Hessian of the Lagrangian
        # (not an approximation of it).

        # Some shortcuts for convenience
        Im = self.Im
        H = self.hess(x,y) ; J = self.nlp.jac(x)
        self.K = BlockLinearOperator([[H, J.T], [-delta*Im]],
                                               symmetric=True)
        return

    def shift_rhs(self, K, delta, rhs):
        n = K[0,0].shape[0]
        shifted_rhs = rhs[:n]+K[0,1]*rhs[n:]/delta
        return shifted_rhs

    def get_dxy(self, y, M_inv, A, b, delta, rhs):
        n=b.shape[0]
        g = rhs[n:]
        dx = M_inv*(b-A*y)
        dy = -y.copy()+g/delta
        return dx, dy

    def solveSystem(self, x, y, rho, delta, rhs, itref_threshold=1.0e-5, nitrefmax=5, **kwargs):
        m=y.shape[0]
        shifted_rhs = self.shift_rhs(self.K,delta,rhs)
        A = self.K[0,1]

        self.log.debug('Solving linear system')
        solver = self.iterative_solver(A)
        y_tilde, istop, itn, normr, normar, normA, condA, normx = solver.solve(shifted_rhs, M=self.HinvOp(),N=1./delta*IdentityOperator(m),
                                                     atol=self.reltol, btol=self.reltol, show=False, **kwargs)

        dx, dy = self.get_dxy(y_tilde, self.HinvOp(), A, shifted_rhs, delta, rhs)

        # solver = self.iterative_solver(self.K, atol=self.reltol, btol=self.reltol)
        # n=x.shape[0]
        # self.log.debug('Solving linear system')
        # step, istop, itn, normr, normar, normA, condA, normx = solver.solve(rhs, show=True, **kwargs)
        # dx1 = step[:n]
        # dy1 = -step[n:]
        # print 'dx:',dx,'dx1:',dx1
        # print 'dy:',dy,'dy1:',dy1
        return istop,'', '',  dx , dy

    def PostIteration(self, x, g, **kwargs):
        """
        This method resets the limited-memory quasi-Newton Hessian after
        every outer iteration.
        """
        s = x - self.x_old
        y = g - self.g_old
        self.Hessapp.store(s, y)
        self.HessInvApp.store(s,y)
        self.x_old = x.copy()
        self.g_old = g.copy()


    def PostInnerIteration(self, x, g, **kwargs):
        """
        This method updates the limited-memory quasi-Newton Hessian by appending
        the most recent (s,y) pair to it and possibly discarding the oldest one
        if all the memory has been used.
        """
        s = x - self.x_old
        y = g - self.g_old
        self.Hessapp.store(s, y)
        self.HessInvApp.store(s,y)

        self.x_old = x.copy()
        self.g_old = g.copy()


def test_merit_function(solver, x,y,dx,dy,rho, delta):
        xTrial = x + dx ; yTrial = y + dy
        phi = solver.phi(x, y, rho, delta, x)
        phiTrial = solver.phi(xTrial, yTrial, rho, delta, x)
        g = solver.dphi(x, y, rho, delta, x)
        print 'x:        ',x       , '  y:       ',y,'  rho:', rho, ' delta:', delta
        print 'dx:      ',dx     , '  dy:     ', dy
        print 'xTrial: ',xTrial, '  yTrial:', yTrial
        print 'phi at x', phi
        print 'phi at xTrial: ', phiTrial
        print 'dphi at x: ', g
        return


def test_RegSQPSolver(filename):
    nlp = AmplModel(filename)         # Create a model
    # nlp.x0 = np.array([0,np.sqrt(3)])
    solver = RegSQPSolver(nlp)
    solver.solve()
    print 'x:', solver.x
    print solver.status
    nlp.close()                              # Close connection with model


def test_RegSQPBFGSIterativeSolver(filename):
    nlp = MFAmplModel(filename)         # Create a model
    # nlp.x0 = np.array([0,np.sqrt(3)])
    IterSolver = LSMRFramework
    solver = RegSQPBFGSIterativeSolver(nlp,IterSolver)
    solver.solve()
    print 'x:', solver.x
    print solver.status
    nlp.close()                              # Close connection with model

if __name__ == '__main__' :

    # Create root logger.
    log = logging.getLogger('regsqp')
    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
    hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)
    log.addHandler(hndlr)

    # Configure the solver logger.
    sublogger = logging.getLogger('regsqp.solver')
    sublogger.setLevel(logging.DEBUG)
    sublogger.addHandler(hndlr)
    sublogger.propagate = False

    filename='/Users/syarra/data/CuteExamples/nl_folder/hs007.nl'
    test_RegSQPSolver(filename)

    test_RegSQPBFGSIterativeSolver(filename)

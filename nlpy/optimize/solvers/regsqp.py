# -*- coding: utf-8 -*-

from pykrylov.linop.blkop import BlockLinearOperator
from pykrylov.linop import IdentityOperator
from pykrylov.minres import Minres as KSolver
from nlpy.model import MFAmplModel
from nlpy.tools.norms import norm2, norm_infty
from nlpy.tools.timing import cputime
try:
    from nlpy.linalg.pyma57 import PyMa57Context as LBLContext
except:
    from nlpy.linalg.pyma27 import PyMa27Context as LBLContext
from pysparse import spmatrix
import numpy as np
import logging

class RegSQPSolver(object):
    """
    A regularized SQP method for degenerate equality-constrained optimization.
    """


    def __init__(self, nlp, **kwargs):
        """
        Instantiate a trust-funnel framework for a given equality-constrained
        problem.

        :keywords:
            :nlp: `NLPModel` instance.
            :abstol: Absolute stopping tolerance
            :reltol: relative required accuracy for || [g-J'y ; c] ||
            :theta: sufficient decrease condition for the inner iterations
            :itermax: maximum number of iterations allowed
            :iterative_solver: solver used to find steps (0=minres, 1=lsmr) #LSMR not supported yet
            :maxit_refine: maximum number of iterative refinements per solve
        """
        self.nlp = nlp
        n = self.nlp.n
        m = self.nlp.m
        self.x = nlp.x0.copy()
        self.y = nlp.pi0.copy()

        self.abstol = kwargs.get('abstol', 1.0e-6)
        self.reltol = kwargs.get('reltol', 1.0e-8)
        self.theta = kwargs.get('theta',0.99)
        self.itermax = kwargs.get('itermax', max(100, 10*nlp.n))
        self.direct = kwargs.get('direct', True)
        self.iterative_solver = kwargs.get('iterative_solver', 0)
        self.krylovtol = kwargs.get('krylovtol', 1.0e-6)

        # Grab logger if one was configured.
        logger_name = kwargs.get('logger_name', 'regsqp.solver')
        self.log = logging.getLogger(logger_name)
        self.verbose = kwargs.get('verbose', True)

        # Create some DiagonalOperators and save them.
        self.In = IdentityOperator(n)
        self.Im = IdentityOperator(m)

        # Set regularization parameters.
        self.rho = kwargs.get('rho', 1.0) ; self.rho_min = 1.0e-8
        self.delta = kwargs.get('delta', 1.0) ; self.delta_min = 1.0e-8

        # Check input parameters.
        if self.rho < 0.0: self.rho = 0.0
        if self.delta < 0.0: self.delta = 0.0

        # Initialize coefficient matrix.
        self.K = self.initialize_coefficient_matrix()

        # System solve method.
        if self.direct:
            Knp = self.K.to_array()
            K = np_to_ll(Knp)
            self.LBL = LBLContext(K, factorize=False, sqd=True)
        self.krylov_solver = KSolver(self.K, reltol=self.krylovtol, logger=self.log)

        # Initialize format strings for display
        fmt_hdr = '%-4s  %9s' + '  %-8s'*3
        self.header = fmt_hdr % ('Iter', 'Obj', 'Delta', 'Rho', 'rNorm')
        self.format  = '%-4d  %9.2e' + '  %-8.2e' * 3

#        self.mu_history = []
#        self.cond_history = []
#        self.berr_history = []
#        self.derr_history = []
#        self.nrms_history = []
#        self.lres_history = []

#        self.condest = kwargs.get('condest', False)
#        self.condest_history = []
#        self.normest_history = []

#        if self.verbose: self.display_stats()

        return

    def initialize_coefficient_matrix(self):
        # [ H+ρI    J' ] [∆x] = [ -g + J'y ]
        # [    J     -δI ] [∆y]    [ -c       ]
        #
        # For now H is the exact Hessian of the Lagrangian
        # (not an approximation of it).

        # Some shortcuts for convenience
        In = self.In ; Im = self.Im
        x0 = self.nlp.x0; rho = self.rho ; delta = self.delta
        H = self.nlp.hess(x0) ; J = self.nlp.jac(x0)
        K = BlockLinearOperator([[H+rho*In, J.T], [-delta*Im]],
                                                      symmetric=True)
        return K

    def update_linear_system(self, x, y, rho, delta, **kwargs):
        # Some shortcuts for convenience
        In = self.In ; Im = self.Im
        H = self.nlp.hess(x, y) ; J = self.nlp.jac(x)
        self.K[0, 0] = H + rho*In
        self.K[0, 1] = J.T
        self.K[1, 1] = -delta*Im
        return

    def initialize_rhs(self):
        return np.zeros(self.nlp.n+self.nlp.m)

    def update_rhs(self, rhs, g, J, y, c):
        n = self.nlp.n
        rhs[:n] = -g + J.T*y
        rhs[n:] = -c
        return

    def phi(self, x, y, rho, delta, x0):
        nlp = self.nlp
        c = nlp.cons(x)
        phi = nlp.obj(x) - np.dot(c,y) + 0.5*rho*norm2(x-x0)**2 + 0.5/delta*norm2(c)**2
        return phi

    def dphi(self, x, y, rho, delta, x0):
        nlp = self.nlp ; n = nlp.n ; m = nlp.m
        dphi = np.zeros(n+m)
        c = nlp.cons(x) ; J = nlp.jac(x)
        print J.to_array()
        dphi[:n] = nlp.grad(x) - J.T*(y-c/delta) + rho*(x-x0)
        dphi[n:] = -c
        return dphi

    def backtracking_linesearch(self, x, y, rho, delta,
                                                 step, bkmax=5, armijo=1.0-4):
        """
        Perform a simple backtracking linesearch on the merit function
        from `x` along `step`.

        Return (new_x, new_y, phi, steplength), where `new_x = x + steplength * step`
        satisfies the Armijo condition and `phi` is the merit function value at
        this new point.
        """
        (dx, dy) = self.get_dxy(step)
        xTrial = x + dx ; yTrial = y + dy
        phi = self.phi(x, y, rho, delta, x)
        phiTrial = self.phi(xTrial, yTrial, rho, delta, x)
        g = self.dphi(x, y, rho, delta, x)
        slope = np.dot(g, step)

        bk = 0
        alpha = 1.0
        while bk < bkmax and \
                phiTrial >= phi + armijo * alpha * slope:
            bk = bk + 1
            alpha /= 1.2
            xTrial = x + alpha * dy ; yTrial = y + alpha * dy
            phiTrial = self.phi(xTrial, yTrial, rho, delta, x)
        return (xTrial, yTrial, phiTrial, alpha)

    def solveSystem(self, rhs, itref_threshold=1.0e-5, nitrefmax=5, **kwargs):

        self.log.debug('Solving linear system')

        if self.direct:
            # Convert Numpy matrix to ll_mat for pyMA27 or pyMA57
            Knp = self.K.to_array()
            K = np_to_ll(Knp)

            # Factorize system.
            self.LBL.factorize(K)

            # TODO:
            # Add a correction of the inertia, by increasing the regularization
            # parameters.

            self.LBL.solve(rhs)
            self.LBL.refine(rhs, tol=itref_threshold, nitref=nitrefmax)
            return self.LBL.x
        else:
            self.krylov_solver.solve(rhs, **kwargs)
            return self.krylov_solver.bestSolution

    def test_solveSystem(self):

        rhs = np.sum(self.K.to_array(), axis=1)
        step = self.solveSystem(rhs, itref_threshold=1.0e-5, nitrefmax=5)
        print 'Yeah'
        print step
        return

    def solve(self, **kwargs):

        # Transfer pointers for convenience.
        itermax = self.itermax ; abstol = self.abstol
        rho = self.rho ; delta = self.delta
        rho_min = self.rho_min ; delta_min = self.delta_min
        nlp = self.nlp ; x = self.x ; y = self.y
        theta = self.theta

        # Get initial objective value
        self.f0 = nlp.obj(x)

        # Allocate room for right-hand side of linear systems.
        rhs = self.initialize_rhs()

        g = nlp.grad(x)
        J = nlp.jac(x)
        c = nlp.cons(x)
        optimal = ( max(norm_infty(g-J.T*y),norm_infty(c))<= abstol)

        finished = False
        iter = 0

        setup_time = cputime()

        finished = False or optimal

        # Main loop.
        while not finished:

            # Step 2
            if delta > 0:
                delta = delta/10
                delta = max(delta, delta_min)
            if rho > 0:
                rho = rho/10
                rho = max(rho, rho_min)

            self.update_linear_system(x, y, rho, delta, **kwargs)
            self.update_rhs(rhs, g, J, y, c)



            step = self.solveSystem(rhs)
            (dx, dy) = self.get_dxy(step)

            # Step 3
            epsilon = 10*delta # a better way to set epsilon dynamically


            # Step 4: Inner Iterations
            xTrial = x + dx ; yTrial = y + dy
            gTrial = nlp.grad(xTrial) ; JTrial = nlp.jac(xTrial)
            cTrial = nlp.cons(xTrial)
            Fnorm = max(norm_infty(g-J.T*y),norm_infty(c))
            FTrialnorm = max(norm_infty(gTrial-JTrial.T*yTrial),norm_infty(cTrial))

            while FTrialnorm > theta*Fnorm + epsilon:

                # Step 3: Compute a new direction p_j
                self.update_linear_system(xTrial, yTrial, rho, delta, **kwargs)
                self.update_rhs(rhs, gTrial, JTrial, yTrial, cTrial)

                stepTrial = self.solveSystem(rhs)

                # Step 4: Backtracking a la Armijo
                (xTrial, yTrial, phi_kj, alpha) = self.backtracking_linesearch(xTrial, yTrial,
                                                                         rho, delta, stepTrial)

                #xTrial = x_kj.copy() ; yTrial = y_kj.copy()
                gTrial = nlp.grad(xTrial) ; JTrial = nlp.jac(xTrial)
                cTrial = nlp.cons(xTrial)
                FTrialnorm = max(norm_infty(gTrial-JTrial.T*yTrial),norm_infty(cTrial))


            # Update values of the new iterate and compute stopping criterion.
            x = xTrial.copy()
            y = yTrial.copy()
            g = nlp.grad(x)
            J = nlp.jac(x)
            c = nlp.cons(x)
            optimal = ( max(norm_infty(g-J.T*y),norm_infty(c))<= abstol)


            # Display initial header every so often.
            if iter % 50 == 0:
                self.log.info(self.header)
                self.log.info('-' * len(self.header))

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

            iter += 1

        solve_time = cputime() - setup_time

        # Transfer final values to class members.
        self.x = xTrial
        self.y = yTrial
        self.f = nlp.obj(self.x)
        self.iter = iter
        self.solve_time = solve_time
        self.status = status
        self.short_status = short_status
        print 'x:', self.x
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

def np_to_ll(npMat):
    llMat = spmatrix.ll_mat_sym(npMat.shape[0])
    for i, row in enumerate(npMat):
        for j, val in enumerate(row[:i+1]):
            if val != 0.0:
                llMat[i, j] = float(val)
    return llMat

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

if __name__ == '__main__' :

    filename='/Users/syarra/data/CuteExamples/nl_folder/hs006.nl'
    nlp = MFAmplModel(filename)         # Create a model
    solver = RegSQPSolver(nlp)

    # Create the coefficient matrix
    solver.initialize_coefficient_matrix()
    print solver.K.to_array()

    print np_to_ll(solver.K.to_array())

    solver.test_solveSystem()

    solver.direct = False
    solver.update_linear_system(nlp.x0, nlp.pi0, 2, 2)
    print solver.K.to_array()
    solver.test_solveSystem()

    x=np.array([1.,0.]) ; y=np.array([2.]) ;
    dx=np.array([0.2,0.2]) ; dy=np.array([0.3])
    delta=10 ; rho=5
    test_merit_function(solver, x,y,dx,dy,rho,delta)
    solver.direct = False
    solver.solve()
    print solver.status

    nlp.close()                              # Close connection with model

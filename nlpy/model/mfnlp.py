"""
mfnlp.py

An abstract class of a matrix-free NLP model for NLPy.
"""


__docformat__ = 'restructuredtext'

# =============================================================================
# Standard Python modules
# =============================================================================
import hashlib

# =============================================================================
# External Python modules
# =============================================================================
import numpy as np

# =============================================================================
# Extension modules
# =============================================================================
from nlpy.model import NLPModel
from nlpy.krylov import SimpleLinearOperator
from nlpy.tools import List


class MFModel(NLPModel):

    """
    MFModel is a derived type of NLPModel which focuses on matrix-free
    implementations of the standard NLP.

    Most of the functionality is the same as NLPModel except for additional
    methods and counters for Jacobian-free cases.

    Note: these parts could be reintegrated into NLPModel at a later date.
    """

    def __init__(self, n=0, m=0, name='Generic Matrix-Free', **kwargs):

        # Standard NLP initialization
        NLPModel.__init__(self,n=n,m=m,name=name,**kwargs)


    def jac(self, x, **kwargs):
        return SimpleLinearOperator(self.n, self.m, symmetric=False,
                         matvec=lambda u: self.jprod(x,u,**kwargs),
                         matvec_transp=lambda u: self.jtprod(x,u,**kwargs))


    def hess(self, x, z=None, **kwargs):
        return SimpleLinearOperator(self.n, self.n, symmetric=True,
                         matvec=lambda u: self.hprod(x,z,u,**kwargs))
# end class



class SlackNLP( NLPModel ):
    """
    General framework for converting a nonlinear optimization problem to a
    form using slack variables.

    In the latter problem, the only inequality constraints are bounds on
    the slack variables. The other constraints are (typically) nonlinear
    equalities.

    The order of variables in the transformed problem is as follows:

    1. x, the original problem variables.

    2. sL = [ sLL | sLR ], sLL being the slack variables corresponding to
       general constraints with a lower bound only, and sLR being the slack
       variables corresponding to the 'lower' side of range constraints.

    3. sU = [ sUU | sUR ], sUU being the slack variables corresponding to
       general constraints with an upper bound only, and sUR being the slack
       variables corresponding to the 'upper' side of range constraints.

    4. tL = [ tLL | tLR ], tLL being the slack variables corresponding to
       variables with a lower bound only, and tLR being the slack variables
       corresponding to the 'lower' side of two-sided bounds.

    5. tU = [ tUU | tUR ], tUU being the slack variables corresponding to
       variables with an upper bound only, and tLR being the slack variables
       corresponding to the 'upper' side of two-sided bounds.

    This framework initializes the slack variables sL, sU, tL, and tU to
    zero by default.

    Note that the slack framework does not update all members of AmplModel,
    such as the index set of constraints with an upper bound, etc., but
    rather performs the evaluations of the constraints for the updated
    model implicitly.
    """

    def __init__(self, nlp, keep_variable_bounds=False, **kwargs):

        self.nlp = nlp
        self.keep_variable_bounds = keep_variable_bounds

        # Save number of variables and constraints prior to transformation
        self.original_n = nlp.n
        self.original_m = nlp.m

        # Number of slacks for inequality constraints with a lower bound
        n_con_low = nlp.nlowerC + nlp.nrangeC ; self.n_con_low = n_con_low

        # Number of slacks for inequality constraints with an upper bound
        n_con_up = nlp.nupperC + nlp.nrangeC ; self.n_con_up = n_con_up

        # Number of slacks for variables with a lower bound
        n_var_low = nlp.nlowerB + nlp.nrangeB ; self.n_var_low = n_var_low

        # Number of slacks for variables with an upper bound
        n_var_up = nlp.nupperB + nlp.nrangeB ; self.n_var_up = n_var_up

        # Update effective number of variables and constraints
        if keep_variable_bounds==True:
            n = self.original_n + n_con_low + n_con_up
            m = self.original_m + nlp.nrangeC
            Lvar = np.zeros(n)
            Lvar[:self.original_n] = nlp.Lvar
            Uvar = +np.infty * np.ones(n)
            Uvar[:self.original_n] = nlp.Uvar

        else:
            n = self.original_n + n_con_low + n_con_up + n_var_low + n_var_up
            m = self.original_m + nlp.nrangeC + n_var_low + n_var_up
            Lvar = np.zeros(n)
            Uvar = +np.infty * np.ones(n)

        Lcon = Ucon = np.zeros(m)

        NLPModel.__init__(self, n=n, m=m, name='Slack-' + nlp.name,
                               Lvar=Lvar, Uvar=Uvar, Lcon=Lcon, Ucon=Ucon)

        # Redefine primal and dual initial guesses
        self.original_x0 = nlp.x0[:]
        self.x0 = np.zeros(self.n)
        self.x0[:self.original_n] = self.original_x0[:]

        self.original_pi0 = nlp.pi0[:]
        self.pi0 = np.zeros(self.m)
        self.pi0[:self.original_m] = self.original_pi0[:]

        # Define index sets for each slack type.
        on = self.original_n
        self.sLL = range(on, on + nlp.nlowerC) ; base = on + nlp.nlowerC
        self.sLR = range(base, base + nlp.nrangeC) ; base += nlp.nrangeC
        self.sUU = range(base, base + nlp.nupperC) ; base += nlp.nupperC
        self.sUR = range(base, base + nlp.nrangeC) ; base += nlp.nrangeC

        if keep_variable_bounds==False:
            self.tLL = range(base, base + nlp.nlowerB) ; base += nlp.nlowerB
            self.tLR = range(base, base + nlp.nrangeB) ; base += nlp.nrangeB
            self.tUU = range(base, base + nlp.nupperB) ; base += nlp.nupperB
            self.tUR = range(base, base + nlp.nrangeB) ; base += nlp.nrangeB

        self.nnzj = nlp.nnzj
        self.nnzh = nlp.nnzh

        # Saved values (private).
        self._last_x = np.infty * np.ones(self.original_n,'d')
        self._last_x_hash = hashlib.sha1(self._last_x).hexdigest()
        self._last_obj = None
        self._last_cons = None
        self._last_grad = None

        return


    def InitializeSlacks(self, val=0.0, **kwargs):
        """
        Initialize all slack variables to given value. This method may need to
        be overridden.
        """
        on = self.original_n

        # Initialize all slacks to given value.
        if val is not None:
            self.x0[self.on:] = val
            return

        # Initialize slacks corresponding to constraints to max(0, c-l) and max(0, u-c).
        x0 = self.original_x0
        c = self.nlp.cons(x0)
        self.x0[self.sLL] = c[self.nlp.lowerC] - self.nlp.Lcon[self.nlp.lowerC]
        self.x0[self.sLR] = c[self.nlp.rangeC] - self.nlp.Lcon[self.nlp.rangeC]
        self.x0[self.sUU] = self.nlp.Ucon[self.nlp.upperC] - c[self.nlp.upperC]
        self.x0[self.sUR] = self.nlp.Ucon[self.nlp.rangeC] - c[self.nlp.rangeC]

        # Initialize slacks corresponding to bounds to max(0, x-l) and max(0, u-x).
        self.x0[self.tLL] = x0[self.nlp.lowerB] - self.nlp.Lvar[self.nlp.lowerB]
        self.x0[self.tLR] = x0[self.nlp.rangeB] - self.nlp.Lvar[self.nlp.rangeB]
        self.x0[self.tUU] = self.nlp.Uvar[self.nlp.upperB] - x0[self.nlp.upperB]
        self.x0[self.tUR] = self.nlp.Uvar[self.nlp.rangeB] - x0[self.nlp.rangeB]

        self.x0[on:] = np.maximum(0, self.x0[on:])
        return


    def obj(self, x):
        """
        Return the value of the objective function at `x`. This function is
        specialized since the original objective function only depends on a
        subvector of `x`.
        """

        x_hash = hashlib.sha1(x[:self.original_n]).hexdigest()
        same_x = self._last_x_hash == x_hash
        # same_x = (self._last_x == x[:self.original_n]).all()

        if self._last_obj is not None and same_x:
            f = self._last_obj
        elif self._last_obj is None and same_x:
            f = self.nlp.obj(self._last_x)
            self._last_obj = f
        else:
            f = self.nlp.obj(x[:self.original_n])
            self._last_obj = f
            self._last_x = x[:self.original_n].copy()
            self._last_x_hash = x_hash
            self._last_cons = None
            self._last_grad = None

        return f


    def grad(self, x):
        """
        Return the value of the gradient of the objective function at `x`.
        This function is specialized since the original objective function only
        depends on a subvector of `x`.
        """
        g = np.zeros(self.n)

        x_hash = hashlib.sha1(x[:self.original_n]).hexdigest()
        same_x = self._last_x_hash == x_hash
        # same_x = (self._last_x == x[:self.original_n]).all()

        if self._last_grad is not None and same_x:
            g[:self.original_n] = self._last_grad
        elif self._last_grad is None and same_x:
            g[:self.original_n] = self.nlp.grad(self._last_x)
            self._last_grad = g[:self.original_n].copy()
        else:
            g[:self.original_n] = self.nlp.grad(x[:self.original_n])
            self._last_obj = None
            self._last_x = x[:self.original_n].copy()
            self._last_x_hash = x_hash
            self._last_cons = None
            self._last_grad = g[:self.original_n].copy()

        return g


    def cons(self, x):
        """
        Evaluate the vector of general constraints for the modified problem.
        Constraints are stored in the order in which they appear in the
        original problem. If constraint i is a range constraint, c[i] will
        be the constraint that has the slack on the lower bound on c[i].
        The constraint with the slack on the upper bound on c[i] will be stored
        in position m + k, where k is the position of index i in
        rangeC, i.e., k=0 iff constraint i is the range constraint that
        appears first, k=1 iff it appears second, etc.

        Constraints appear in the following order:

        1. [ c  ]   general constraints in original order
        2. [ cR ]   'upper' side of range constraints
        3. [ b  ]   linear constraints corresponding to bounds on original problem
        4. [ bR ]   linear constraints corresponding to 'upper' side of two-sided
                    bounds
        """
        n = self.n ; on = self.original_n
        m = self.m ; om = self.original_m
        nlp = self.nlp

        equalC = nlp.equalC
        lowerC = nlp.lowerC ; nlowerC = nlp.nlowerC
        upperC = nlp.upperC ; nupperC = nlp.nupperC
        rangeC = nlp.rangeC ; nrangeC = nlp.nrangeC

        c = np.empty(m)
        x_hash = hashlib.sha1(x[:self.original_n]).hexdigest()
        same_x = self._last_x_hash == x_hash
        # same_x = (self._last_x == x[:self.original_n]).all()
        if self._last_cons is not None and same_x:
            c[:om] = self._last_cons
        elif self._last_cons is None and same_x:
            c[:om] = nlp.cons(self._last_x)
            self._last_cons = c[:om].copy()
        else:
            c[:om] = nlp.cons(x[:on])
            self._last_obj = None
            self._last_x = x[:self.original_n].copy()
            self._last_x_hash = x_hash
            self._last_cons = c[:om].copy()
            self._last_grad = None

        c[om:om+nrangeC] = c[rangeC]

        c[equalC] -= nlp.Lcon[equalC]
        c[lowerC] -= nlp.Lcon[lowerC] ; c[lowerC] -= x[self.sLL]

        c[upperC] -= nlp.Ucon[upperC] ; c[upperC] *= -1.
        c[upperC] -= x[self.sUU]

        c[rangeC] -= nlp.Lcon[rangeC] ; c[rangeC] -= x[self.sLR]

        c[om:om+nrangeC] -= nlp.Ucon[rangeC]
        c[om:om+nrangeC] *= -1
        c[om:om+nrangeC] -= x[self.sUR]

        if self.keep_variable_bounds==False:
            # Add linear constraints corresponding to bounds on original problem
            lowerB = nlp.lowerB ; nlowerB = nlp.nlowerB ; Lvar = nlp.Lvar
            upperB = nlp.upperB ; nupperB = nlp.nupperB ; Uvar = nlp.Uvar
            rangeB = nlp.rangeB ; nrangeB = nlp.nrangeB

            b = c[om+nrangeC:]
            b[:nlowerB] = x[lowerB] - Lvar[lowerB] - x[self.tLL]
            b[nlowerB:nlowerB+nrangeB] = x[rangeB] - Lvar[rangeB] \
                                        - x[self.tLR]
            b[nlowerB+nrangeB:nlowerB+nrangeB+nupperB] = Uvar[upperB] \
                                        - x[upperB] - x[self.tUU]
            b[nlowerB+nrangeB+nupperB:] = Uvar[rangeB] - x[rangeB] \
                                        - x[self.tUR]

        return c


    def Bounds(self, x):
        """
        Evaluate the vector of equality constraints corresponding to bounds
        on the variables in the original problem.
        """
        if keep_variable_bounds == False:
            lowerB = self.lowerB ; nlowerB = self.nlowerB
            upperB = self.upperB ; nupperB = self.nupperB
            rangeB = self.rangeB ; nrangeB = self.nrangeB

            n  = self.n ; on = self.original_n

            b = np.empty(n + nrangeB)
            b[:n] = x[:]
            b[n:] = x[rangeB]

            b[lowerB] -= self.Lvar[lowerB]
            b[upperB] -= self.Uvar[upperB] ; b[upperB] *= -1
            b[rangeB] -= self.Lvar[rangeB]
            b[n:]     -= self.Uvar[rangeB] ; b[n:] *= -1

            b[lowerB] -= x[self.tLL]
            b[upperB] -= x[self.tUU]
            b[rangeB] -= x[self.tLR]
            b[n:]     -= x[self.tUR]
            return b
        else:
            return None


    def jprod(self, x, v, **kwargs):
        n = self.n ; on = self.original_n
        m = self.m ; om = self.original_m
        nlp = self.nlp

        # List() simply allows operations such as 1 + [2,3] -> [3,4]
        lowerC = List(nlp.lowerC) ; nlowerC = nlp.nlowerC
        upperC = List(nlp.upperC) ; nupperC = nlp.nupperC
        rangeC = List(nlp.rangeC) ; nrangeC = nlp.nrangeC
        lowerB = List(nlp.lowerB) ; nlowerB = nlp.nlowerB
        upperB = List(nlp.upperB) ; nupperB = nlp.nupperB
        rangeB = List(nlp.rangeB) ; nrangeB = nlp.nrangeB

        p = np.zeros(m)

        p[:om] = nlp.jprod(x[:on], v[:on])
        p[upperC] *= -1.0
        p[om:om+nrangeC] = p[rangeC]
        p[om:om+nrangeC] *= -1.0

        # Insert contribution of slacks on general constraints
        bot = on;       p[lowerC] -= v[bot:bot+nlowerC]
        bot += nlowerC; p[rangeC] -= v[bot:bot+nrangeC]
        bot += nrangeC; p[upperC] -= v[bot:bot+nupperC]
        bot += nupperC; p[om:om+nrangeC] -= v[bot:bot+nrangeC]

        if self.keep_variable_bounds==False:
            # Insert contribution of bound constraints on the original problem
            print 'lowerB:', lowerB, 'upperB', upperB, 'rangeB', rangeB
            print 'p:', p
            print 'v:', v
            bot = om+nrangeC; p[bot:bot+nlowerB] += v[lowerB]
            print bot
            bot += nlowerB;  p[bot:bot+nrangeB] += v[rangeB]; print bot
            bot += nrangeB;  p[bot:bot+nupperB] -= v[upperB]; print bot
            bot += nupperB;  p[bot:bot+nrangeB] -= v[rangeB]; print bot

            # Insert contribution of slacks on the bound constraints
            bot = om+nrangeC; p[bot:bot+nlowerB] -= v[self.tLL]
            bot += nlowerB;  p[bot:bot+nrangeB] -= v[self.tLR]
            bot += nrangeB;  p[bot:bot+nupperB] -= v[self.tUU]
            bot += nupperB;  p[bot:bot+nrangeB] -= v[self.tUR]

        return p


    def jtprod(self, x, v, **kwargs):
        n = self.n ; on = self.original_n
        m = self.m ; om = self.original_m
        nlp = self.nlp

        # List() simply allows operations such as 1 + [2,3] -> [3,4]
        lowerC = List(nlp.lowerC) ; nlowerC = nlp.nlowerC
        upperC = List(nlp.upperC) ; nupperC = nlp.nupperC
        rangeC = List(nlp.rangeC) ; nrangeC = nlp.nrangeC
        lowerB = List(nlp.lowerB) ; nlowerB = nlp.nlowerB
        upperB = List(nlp.upperB) ; nupperB = nlp.nupperB
        rangeB = List(nlp.rangeB) ; nrangeB = nlp.nrangeB

        p = np.zeros(n)
        print 'p:',p, 'v:', v
        print 'rangeC:', rangeC
        vmp = v[:om].copy()
        vmp[upperC] *= -1.0
        vmp[rangeC] -= v[om:om+nrangeC]

        p[:on] = nlp.jtprod(x[:on], vmp)

        # Insert contribution of slacks on general constraints
        bot = on;       p[on:on+nlowerC]    = -v[lowerC]
        bot += nlowerC; p[bot:bot+nrangeC]  = -v[rangeC]
        bot += nrangeC; p[bot:bot+nupperC]  = -v[upperC]
        bot += nupperC; p[bot:bot+nrangeC]  = -v[om:om+nrangeC]

        if self.keep_variable_bounds==False:
            # Insert contribution of bound constraints on the original problem
            bot = om+nrangeC; p[lowerB] += v[bot:bot+nlowerB]
            bot += nlowerB;  p[rangeB] += v[bot:bot+nrangeB]
            bot += nrangeB;  p[upperB] -= v[bot:bot+nupperB]
            bot += nupperB;  p[rangeB] -= v[bot:bot+nrangeB]

            # Insert contribution of slacks on the bound constraints
            bot = om+nrangeC; p[self.tLL] -= v[bot:bot+nlowerB]
            bot += nlowerB;  p[self.tLR] -= v[bot:bot+nrangeB]
            bot += nrangeB;  p[self.tUU] -= v[bot:bot+nupperB]
            bot += nupperB;  p[self.tUR] -= v[bot:bot+nrangeB]

        return p

    def hprod(self, x, y, v, **kwargs):
        on = self.original_n ; om = self.original_m
        Hv = np.zeros(self.n)
        Hv[:on] = self.nlp.hprod(x[:on], y[:om], v[:on], **kwargs)
        return Hv


class MFSlackNLP( SlackNLP, MFModel ):
    def __init__(self, nlp, **kwargs):

        # Standard initializations

        MFModel.__init__(self,**kwargs)
        SlackNLP.__init__(self,nlp,**kwargs)

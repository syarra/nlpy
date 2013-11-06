"""
snlp.py

An slack framework for NLPy.
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
from nlpy.tools import List
from pysparse.sparse.pysparseMatrix import PysparseMatrix as sp


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

    :parameters:
        :nlp:  Original NLP to transform to a slack form.

    :keywords:
        :keep_variable_bounds: set to `True` if you don't want to convert
                               bounds on variables to inequalities. In this
                               case bounds are kept as they were in the
                               original NLP.

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

        # Number of slacks for the constaints
        nSlacks_s = n_con_low + n_con_up; self.nSlacks_s = nSlacks_s

        # Create some lists
        bot = self.original_n; self.sLL = range(bot, bot + nlp.nlowerC)
        bot += nlp.nlowerC;    self.sLR = range(bot, bot + nlp.nrangeC)
        bot += nlp.nrangeC;    self.sUU = range(bot, bot + nlp.nupperC)
        bot += nlp.nupperC;    self.sUR = range(bot, bot + nlp.nrangeC)

        # Update effective number of variables and constraints
        if keep_variable_bounds==True:
            n = self.original_n + n_con_low + n_con_up
            m = self.original_m + nlp.nrangeC

            Lvar = np.zeros(n)
            Lvar[:self.original_n] = nlp.Lvar
            Uvar = +np.infty * np.ones(n)
            Uvar[:self.original_n] = nlp.Uvar

        else:
            # Create more lists
            bot += nlp.nrangeC;    self.tLL = range(bot, bot + nlp.nlowerB)
            bot += nlp.nlowerB;    self.tLR = range(bot, bot + nlp.nrangeB)
            bot += nlp.nrangeB;    self.tUU = range(bot, bot + nlp.nupperB)
            bot += nlp.nupperB;    self.tUR = range(bot, bot + nlp.nrangeB)

            n = self.original_n + n_con_low + n_con_up + n_var_low + n_var_up
            m = self.original_m + nlp.nrangeC + n_var_low + n_var_up
            Lvar = np.zeros(n)
            Uvar = +np.infty * np.ones(n)
            Lvar[:self.original_n] = -np.infty * np.ones(self.original_n)

        Lcon = Ucon = np.zeros(m)

        NLPModel.__init__(self, n=n, m=m, name='Slack-'+nlp.name, Lvar=Lvar, \
                          Uvar=Uvar, Lcon=Lcon, Ucon=Ucon)

        # Redefine primal and dual initial guesses
        self.original_x0 = nlp.x0[:]
        self.x0 = np.zeros(self.n)
        self.x0[:self.original_n] = self.original_x0[:]

        self.original_pi0 = nlp.pi0[:]
        self.pi0 = np.zeros(self.m)
        self.pi0[:self.original_m] = self.original_pi0[:]

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
        self.x0[self.original_n:] = val
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
        return c


    def Bounds(self, x):
        """
        Evaluate the vector of equality constraints corresponding to bounds
        on the variables in the original problem.
        """

        if self.keep_variable_bounds == False:
            # Create some shortcuts for convenience
            nlp = self.nlp
            n  = self.n ; on = self.original_n
            lowerB = nlp.lowerB ; nlowerB = nlp.nlowerB
            upperB = nlp.upperB ; nupperB = nlp.nupperB
            rangeB = nlp.rangeB ; nrangeB = nlp.nrangeB

            b = np.empty(n + nrangeB)
            b[:n] = x[:]
            b[n:] = x[rangeB]

            b[lowerB] -= self.Lvar[lowerB] ;
            b[lowerB] -= x[self.tLL]

            b[upperB] -= self.Uvar[upperB] ; b[upperB] *= -1
            b[upperB] -= x[self.tUU]

            b[rangeB] -= self.Lvar[rangeB] ; b[rangeB] -= x[self.tLR]
            b[n:]     -= self.Uvar[rangeB] ; b[n:] *= -1
            b[n:]     -= x[self.tUR]
            return b
        else:
            return None


    def jprod(self, x, v, **kwargs):
        """
        Evaluate the Jacobian matrix-vector product of all equality
        constraints of the transformed problem with a vector `v` (J(x).T v).
        See the documentation of :meth:`jac` for more details on how the
        constraints are ordered.
        """
        # Create some shortcuts for convenience
        nlp = self.nlp
        on = self.original_n; om = self.original_m ; n = self.n ; m = self.m
        lowerC = nlp.lowerC ; upperC = nlp.upperC ; rangeC = nlp.rangeC
        nrangeC = nlp.nrangeC

        p = np.zeros(m)

        p[:om] = nlp.jprod(x[:on], v[:on])
        p[upperC] *= -1.0
        p[om:om+nrangeC] = p[rangeC]
        p[om:om+nrangeC] *= -1.0

        # Insert contribution of slacks on general constraints
        p[lowerC] -= v[self.sLL]
        p[rangeC] -= v[self.sLR]
        p[upperC] -= v[self.sUU]
        p[om:om+nrangeC] -= v[self.sUR]

        if self.keep_variable_bounds==False:
            # Create some more shortcuts
            lowerB = nlp.lowerB ; upperB = nlp.upperB ; rangeB = nlp.rangeB
            nlowerB = nlp.nlowerB ; nupperB = nlp.nupperB ; nrangeB = nlp.nrangeB

            # Insert contribution of bound constraints on the original problem
            bot = om+nrangeC; p[bot:bot+nlowerB] += v[lowerB]
            bot += nlowerB;   p[bot:bot+nrangeB] += v[rangeB]
            bot += nrangeB;   p[bot:bot+nupperB] -= v[upperB]
            bot += nupperB;   p[bot:bot+nrangeB] -= v[rangeB]

            # Insert contribution of slacks on the bound constraints
            bot = om+nrangeC; p[bot:bot+nlowerB] -= v[self.tLL]
            bot += nlowerB;   p[bot:bot+nrangeB] -= v[self.tLR]
            bot += nrangeB;   p[bot:bot+nupperB] -= v[self.tUU]
            bot += nupperB;   p[bot:bot+nrangeB] -= v[self.tUR]
        return p


    def jtprod(self, x, v, **kwargs):
        """
        Evaluate the Jacobian-transpose matrix-vector product of all equality
        constraints of the transformed problem with a vector `v` (J(x).T v).
        See the documentation of :meth:`jac` for more details on how the
        constraints are ordered.
        """
        # Create some shortcuts for convenience
        nlp = self.nlp
        on = self.original_n ; om = self.original_m ; n = self.n ; m = self.m
        lowerC = nlp.lowerC ; upperC = nlp.upperC ; rangeC = nlp.rangeC
        nrangeC = nlp.nrangeC;

        p = np.zeros(n)
        vmp = v[:om].copy()
        vmp[upperC] *= -1.0
        vmp[rangeC] -= v[om:om+nrangeC]

        p[:on] = nlp.jtprod(x[:on], vmp)

        # Insert contribution of slacks on general constraints
        p[self.sLL] = -v[lowerC]
        p[self.sLR] = -v[rangeC]
        p[self.sUU] = -v[upperC]
        p[self.sUR] = -v[om:om+nrangeC]

        if self.keep_variable_bounds==False:
            # Create some more shortcuts
            lowerB = nlp.lowerB ; upperB = nlp.upperB ; rangeB = nlp.rangeB
            nlowerB = nlp.nlowerB ; nupperB = nlp.nupperB ; nrangeB = nlp.nrangeB

            # Insert contribution of bound constraints on the original problem
            bot = om+nrangeC; p[lowerB] += v[bot:bot+nlowerB]
            bot += nlowerB;   p[rangeB] += v[bot:bot+nrangeB]
            bot += nrangeB;   p[upperB] -= v[bot:bot+nupperB]
            bot += nupperB;   p[rangeB] -= v[bot:bot+nrangeB]

            # Insert contribution of slacks on the bound constraints
            bot = om+nrangeC; p[self.tLL] -= v[bot:bot+nlowerB]
            bot += nlowerB;   p[self.tLR] -= v[bot:bot+nrangeB]
            bot += nrangeB;   p[self.tUU] -= v[bot:bot+nupperB]
            bot += nupperB;   p[self.tUR] -= v[bot:bot+nrangeB]
        return p


    def _jac(self, x, lp=False):
        """
        Helper method to assemble the Jacobian matrix of the constraints of the
        transformed problems. See the documentation of :meth:`jac` for more
        information.

        The positional argument `lp` should be set to `True` only if the problem
        is known to be a linear program. In this case, the evaluation of the
        constraint matrix is cheaper and the argument `x` is ignored.
        """
        n = self.n ; m = self.m ; nlp = self.nlp
        on = self.original_n ; om = self.original_m

        # List() simply allows operations such as 1 + [2,3] -> [3,4]
        lowerC = List(nlp.lowerC) ; nlowerC = nlp.nlowerC
        upperC = List(nlp.upperC) ; nupperC = nlp.nupperC
        rangeC = List(nlp.rangeC) ; nrangeC = nlp.nrangeC
        lowerB = List(nlp.lowerB) ; nlowerB = nlp.nlowerB
        upperB = List(nlp.upperB) ; nupperB = nlp.nupperB
        rangeB = List(nlp.rangeB) ; nrangeB = nlp.nrangeB
        nbnds  = nlowerB + nupperB + 2*nrangeB
        nSlacks = nlowerC + nupperC + 2*nrangeC

        # Initialize sparse Jacobian
        nnzJ = 2 * self.nlp.nnzj + m + nrangeC + nbnds + nrangeB  # Overestimate
        J = sp(nrow=m, ncol=n, sizeHint=nnzJ)

        # Insert contribution of general constraints
        if lp:
            J[:om,:on] = self.nlp.A()
        else:
            J[:om,:on] = self.nlp.jac(x[:on])
        J[upperC,:on] *= -1.0               # Flip sign of 'upper' gradients
        J[om:om+nrangeC,:on] = J[rangeC,:on]  # Append 'upper' side of range const.
        J[om:om+nrangeC,:on] *= -1.0        # Flip sign of 'upper' range gradients.

        # Create a few index lists
        rlowerC = List(range(nlowerC)) ; rlowerB = List(range(nlowerB))
        rupperC = List(range(nupperC)) ; rupperB = List(range(nupperB))
        rrangeC = List(range(nrangeC)) ; rrangeB = List(range(nrangeB))

        # Insert contribution of slacks on general constraints
        J.put(-1.0,      lowerC,  on + rlowerC)
        J.put(-1.0,      upperC,  on + nlowerC + rupperC)
        J.put(-1.0,      rangeC,  on + nlowerC + nupperC + rrangeC)
        J.put(-1.0, om + rrangeC, on + nlowerC + nupperC + nrangeC + rrangeC)

        if self.keep_variable_bounds==False:
            # Insert contribution of bound constraints on the original problem
            bot  = om+nrangeC ; J.put( 1.0, bot + rlowerB, lowerB)
            bot += nlowerB    ; J.put( 1.0, bot + rrangeB, rangeB)
            bot += nrangeB    ; J.put(-1.0, bot + rupperB, upperB)
            bot += nupperB    ; J.put(-1.0, bot + rrangeB, rangeB)

            # Insert contribution of slacks on the bound constraints
            bot  = om+nrangeC
            J.put(-1.0, bot + rlowerB, on + nSlacks + rlowerB)
            bot += nlowerB
            J.put(-1.0, bot + rrangeB, on + nSlacks + nlowerB + rrangeB)
            bot += nrangeB
            J.put(-1.0, bot + rupperB, on + nSlacks + nlowerB + nrangeB + rupperB)
            bot += nupperB
            J.put(-1.0, bot + rrangeB, on+nSlacks+nlowerB+nrangeB+nupperB+rrangeB)
        return J


    def jac(self, x):
        """
        Evaluate the Jacobian matrix of all equality constraints of the
        transformed problem. The gradients of the general constraints appear in
        'natural' order, i.e., in the order in which they appear in the problem.
        The gradients of range constraints appear in two places: first in the
        'natural' location and again after all other general constraints, with a
        flipped sign to account for the upper bound on those constraints.

        The gradients of the linear equalities corresponding to bounds on the
        original variables appear in the following order:

        1. variables with a lower bound only
        2. lower bound on variables with two-sided bounds
        3. variables with an upper bound only
        4. upper bound on variables with two-sided bounds

        The overall Jacobian of the new constraints thus has the form::

            [ J    -I         ]
            [-JR      -I      ]
            [ I          -I   ]
            [-I             -I]

        where the columns correspond, in order, to the variables `x`, `s`, `sU`,
        `t`, and `tU`, the rows correspond, in order, to

        1. general constraints (in natural order)
        2. 'upper' side of range constraints
        3. bounds, ordered as explained above
        4. 'upper' side of two-sided bounds,

        and where the signs corresponding to 'upper' constraints and upper
        bounds are flipped in the (1,1) and (3,1) blocks.
        """
        return self._jac(x, lp=False)


    def A(self):
        """
        Return the constraint matrix if the problem is a linear program. See the
        documentation of :meth:`jac` for more information.
        """
        return self._jac(0, lp=True)


    def hprod(self, x, y, v, **kwargs):
        """
        Evaluate the Hessian vector product of the Lagrangian.
        """
        if y is None: y = np.zeros(self.m)
        # Create some shortcuts for convenience
        nlp = self.nlp
        on = self.original_n ; om = self.original_m
        upperC = nlp.upperC ; nupperC = nlp.nupperC
        rangeC = nlp.rangeC ; nrangeC = nlp.nrangeC

        Hv = np.zeros(self.n)
        pi = y[:om].copy()
        pi[upperC] *= -1.
        pi[rangeC] -= y[om:om+nrangeC].copy()

        Hv[:on] = self.nlp.hprod(x[:on], pi, v[:on], **kwargs)
        return Hv


    def hess(self, x, z=None, obj_num=0, *args, **kwargs):
        """
        Evaluate the Hessian of the Lagrangian.
        """
        if z is None: z = np.zeros(self.m)
        # Create some shortcuts for convenience
        nlp = self.nlp
        on = self.original_n ; om = self.original_m
        upperC = nlp.upperC ; nupperC = nlp.nupperC
        rangeC = nlp.rangeC ; nrangeC = nlp.nrangeC

        pi = z[:om].copy()
        pi[upperC] *= -1.

        pi[rangeC] -= z[om:om+nrangeC].copy()

        H = sp(nrow=self.n, ncol=self.n, symmetric=True, sizeHint=self.nlp.nnzh)
        H[:on, :on] = self.nlp.hess(x[:on], pi, obj_num, *args, **kwargs)
        return H


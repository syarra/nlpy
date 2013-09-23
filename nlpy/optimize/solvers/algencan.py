import pywrapper
import numpy as np
from nlpy.model import MFAmplModel


class AlgencanModel(object):
    def __init__(self, nlp, **kwargs):
        self.nlp = nlp
        return

class AlgencanFramework(object):
    """
    """
    def __init__(self, algencanpb, **kwargs):
        self.nlp = algencanpb
        algencanpb.display_basic_info()
     
    def Solve(self, **kwargs):
        """
        Call algencan solver.
        """
        nlp = self.nlp

        param = {
            'epsfeas': 1.0e-08,
            'epsopt' : 1.0e-08,

            'efacc'  : 1.0e-04,
            'eoacc'  : 1.0e-04,

            'iprint': 10,
            'ncomp' : 6} 


        def inip():
            """
            (From Algencan-2.4.0)
            This subroutine must set some problem data.

            For achieving this objective YOU MUST MODIFY it according to your
            problem. See below where your modifications must be inserted.

            Parameters of the subroutine:

            On Entry:

            This subroutine has no input parameters.

            On Return

            n        integer,
                     number of variables,

            x        double precision x(n),
                     initial point,

            l        double precision l(n),
                     lower bounds on x,

            u        double precision u(n),
                     upper bounds on x,

            m        integer,
                     number of constraints (excluding the bounds),

            lambda   double precision lambda(m),
                     initial estimation of the Lagrange multipliers,

            equatn   logical equatn(m)
                     for each constraint j, set equatn(j) = .true. if it is an
                     equality constraint of the form c_j(x) = 0, and set
                     equatn(j) = .false. if it is an inequality constraint of
                     the form c_j(x) <= 0,

            linear   logical linear(m)
                     for each constraint j, set linear(j) = .true. if it is a
                     linear constraint, and set linear(j) = .false. if it is a
                     nonlinear constraint.
            """
        #   Number of variables

            n = nlp.n

        #   Number of constraints (equalities plus inequalities)

            m = nlp.nequalC + nlp.nlowerC + nlp.nupperC + nlp.nrangeC
            print 'n:',n,'m:',m
        #   Initial point

            x = nlp.x0

        #   Lower and upper bounds

            l = nlp.Lvar
            u = nlp.Uvar
            for i in range(n):
                if l[i]==-np.inf:
                    l[i]=-1e20

                if u[i]==np.inf:
                    u[i]=1e20
            
        #   Lagrange multipliers approximation. Most users prefer to use the
        #   null initial Lagrange multipliers estimates. However, if the
        #   problem that you are solving is "slightly different" from a
        #   previously solved problem of which you know the correct Lagrange
        #   multipliers, we encourage you to set these multipliers as initial
        #   estimates. Of course, in this case you are also encouraged to use
        #   the solution of the previous problem as initial estimate of the
        #   solution. Similarly, most users prefer to use rho = 10 as initial
        #   penalty parameters. But in the case mentioned above (good
        #   estimates of solution and Lagrange multipliers) larger values of
        #   the penalty parameters (say, rho = 1000) may be more useful. More
        #   warm-start procedures are being elaborated.

            lambda_ = np.zeros(m)

        #   For each constraint i, set equatn[i] = 1. if it is an equality
        #   constraint of the form c_i(x) = 0, and set equatn[i] = 0 if
        #   it is an inequality constraint of the form c_i(x) <= 0.

            equatn = [False] * m
            for j in nlp.equalC:
                equatn[j] = True

        #   For each constraint i, set linear[i] = 1 if it is a linear
        #   constraint, otherwise set linear[i] = 0.

            linear = [False] * m
            for j in nlp.lin:
                linear[j] = True

        #   In this Python interface evalf, evalg, evalh, evalc, evaljac and
        #   evalhc are present. evalfc, evalgjac, evalhl and evalhlp are not.

        #   In this interface, we use Algencan with its matrix-free interface,
        #   meaning that we only provide routines to compute matrix-vector products.
            coded = [False,  # evalf
                     False,  # evalg
                     False,  # evalh
                     False,  # evalc
                     False,  # evaljac
                     False,  # evalhc
                     True,   # evalfc
                     False,  # evalgjac
                     True,   # evalgjacp
                     False,  # evalhl
                     True]   # evalhlp

        #   Set checkder = 1 if you code some derivatives and you would
        #   like them to be tested by finite differences. It is highly
        #   recommended.

            checkder = False
            print n,len(x), len(u), len(l),m,len(lambda_),len(equatn),len(linear),len(coded)
            return n,x,l,u,m,lambda_,equatn,linear,coded,checkder

        def evalf(x):
            """
            This subroutine must compute the objective function.
            """
            flag = -1
            f = 0.0
            return f,flag

        def evalg(x):
            """
            This subroutine must compute the gradient vector of the objective
            function.
            """
            flag = -1
            g = zeros(self.nlp.n)
            return g,flag

        def evalh(x):
            """
            This subroutine might compute the Hessian matrix of the objective
            function.
            """
            flag = -1
            hnnz = 0
            hlin = zeros(hnnz, int)
            hcol = zeros(hnnz, int)
            hval = zeros(hnnz, float)
            return hlin,hcol,hval,hnnz,flag

        def evalc(x,ind):
            """
            This subroutine must compute the ind-th constraint.
            """
            flag = -1
            c = 0.0
            return c,flag

        def evaljac(x,ind):
            """
            This subroutine must compute the gradient of the ind-th constraint.
            """
            flag = -1
            jcnnz = 2
            jcvar = zeros(jcnnz, int)
            jcval = zeros(jcnnz, float)
            return jcvar,jcval,jcnnz,flag

        def evalhc(x,ind):
            """
            This subroutine might compute the Hessian matrix of the ind-th
            constraint.
            """
            flag = -1
            hcnnz = 0
            hclin = zeros(hcnnz, int)
            hccol = zeros(hcnnz, int)
            hcval = zeros(hcnnz, float)
            return hclin,hccol,hcval,hcnnz,flag

        def evalfc(x, m):
            flag = 0
            f = nlp.obj(x)
            c = -nlp.consPos(x)
            return f,c,flag

        def evalgjac(x,m):
            flag = -1
            n = len(x)
            g = zeros(n)
            jcnnz = 0
            jcfun = zeros(jcnnz, int)
            jcvar = zeros(jcnnz, int)
            jcval = zeros(jcnnz, float)
            return g,jcfun,jcvar,jcval,jcnnz,flag

        def evalgjacp(x, m, u, work, gotj):
            """
            (From algencan-2.4.0/sources/problems/toyprob3.f)
            The meaning of argument work follows: work = 'j' or 'J' means that 
            q is an input array argument and that p = Jacobian x q must be 
            computed, while work = 't' or 'T' means that p is an input array
            argument and that q = Jacobian^t x p must be computed. Moreover, a 
            capital letter (i.e. 'J' or 'T') means that the gradient of the 
            objective function g must also be computed. A lower letter (i.e. 
            'j' or 't') means that only the product with the Jacobian its 
            required. In the later case, input array argument g MUST NOT be 
            modified nor referenced.
            """
            #print 'evalgjacp'
            flag = 0
            n = len(x)
            p = np.zeros(m)

            if work == "J" or work == "T":
                g = nlp.grad(x)

            if work == "j" or work == "J":
                p = -nlp.jprodPos(x,u)
            elif work == "t" or work == "T":
                p = -nlp.jtprodPos(x,u)

            if work == "j" or work == "t":
                return p, gotj, flag
            elif work == "J" or work == "T":
                return g, p, gotj, flag

        def evalhl(x,m,lambda_,scalef,scalec):
            flag = -1
            hlnnz = 0
            hllin = zeros(hlnnz, int)
            hlcol = zeros(hlnnz, int)
            hlval = zeros(hlnnz, float)
            return hllin,hlcol,hlval,hlnnz,flag

        def evalhlp(x, m, lambda_, scalef, scalec, p, goth):
            flag = 0
            n = len(x)
            hp = self.nlp.hprod(x,-lambda_,p)
            return hp,goth,flag

        def endp(x, l, u, m, lambda_, equatn, linear):
            """
            This subroutine can be used to do some extra job.

            This subroutine can be used to do some extra job after the solver
            has found the solution, like some extra statistics, or to save the
            solution in some special format or to draw some graphical
            representation of the solution. If the information given by the
            solver is enough for you then leave the body of this subroutine
            empty.

            Parameters of the subroutine:

            The parameters of this subroutine are the same parameters of
            subroutine inip. But in this subroutine there are not output
            parameter. All the parameters are input parameters.
            """

            print('algencan: End of Execution')
            print('')
            print('Final variables: %-s' % repr(x))
            print('-------------------------------')
            pass


        pywrapper.solver(evalf,evalg,evalh,evalc,evaljac,evalhc,evalfc,evalgjac,
               evalgjacp,evalhl,evalhlp,inip,endp,param)
        print 'problem solved'



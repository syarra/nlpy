import numpy as np
from nlpy.model import MFModel
from nlpy.model.mfnlp import SlackNLP
from nlpy.krylov.linop import PysparseLinearOperator

def obj(x):
    return np.array([x[0]**2 + x[1]*np.sin(x[0] + x[2]) + 3.* x[1]**4*x[2]**4 + x[1]])

def grad(x):
    g = np.zeros(3)
    g[0] = 2.*x[0] + x[1]*np.cos(x[0]+x[2])
    g[1] = np.sin(x[0]+x[2]) + 12.*x[1]**3*x[2]**4 + 1.
    g[2] = x[1]*np.cos(x[0]+x[2]) + 12.*x[1]**4*x[2]**3
    return g

def hprod(x, pi, v):
    H = np.zeros([3,3])
    H[0,0] = 2. - x[1]*np.sin(x[0]+x[2])
    H[0,1] = np.cos(x[0]+x[2])
    H[0,2] = -x[1]*np.sin(x[0]+x[2])

    H[1,0] = np.cos(x[0]+x[2])
    H[1,1] = 36.*x[1]**2*x[2]**4
    H[1,2] = np.cos(x[0]+x[2]) + 48.*x[1]**3*x[2]**3

    H[2,0] = -x[1]*np.sin(x[0]+x[2])
    H[2,1] = np.cos(x[0]+x[2]) + 48.*x[1]**3*x[2]**3
    H[2,2] = -x[1]*np.sin(x[0]+x[2]) + 36.*x[1]**4*x[2]**3

    return np.dot(H,v)

def cons(x):
    return np.array([np.cos(x[0]+2.*x[1]-1)])

def igrad(i, x):
    g = np.zeros(3)
    g[0] = -np.sin(x[0]+2.*x[1] -1)
    g[1] = -2.*np.sin(x[0]+2.*x[1]-1)
    g[2] = 0.
    return g

def jprod(x, v):
    j = np.zeros([1,3])
    j[0,0] = -np.sin(x[0]+2.*x[1] -1)
    j[0,1] = -2.*np.sin(x[0]+2.*x[1]-1)
    j[0,2] = 0.
    return np.dot(j,v)

def jtprod(x, v):
    j = np.zeros([3,1])
    j[0,0] = -np.sin(x[0]+2.*x[1] -1)
    j[1,0] = -2.*np.sin(x[0]+2.*x[1]-1)
    j[2,0] = 0.
    return np.dot(j,v)

def hiprod(x, i, v):
    H = np.zeros([3,3])
    H[0,0] = -np.cos(x[0]+2.*x[1]-1)
    H[0,1] = -2.*np.cos(x[0]+2.*x[1]-1)
    H[0,2] = 0.

    H[1,0] = -2.*np.cos(x[0]+2.*x[1]-1)
    H[1,1] = -4.*np.cos(x[0]+2.*x[1]-1)
    H[1,2] = 0.

    H[2,0] = H[2,1] = H[2,2] = 0
    return np.dot(H,v)

lvar = np.array([-np.infty, -1., 1.])
uvar = np.array([np.infty, 1., 2.])
x0 = np.array([0.,0.,1.5])
ucon = np.array([np.infty])
lcon = np.array([1.])

nlp = MFModel(n=3,m=1,name='AUGL', Lvar=lvar, Uvar=uvar, Lcon=lcon, Ucon=ucon,
               x0=x0)
nlp.obj = obj
nlp.grad = grad
nlp.hprod = hprod
nlp.cons = cons
nlp.igrad = igrad
nlp.jprod = jprod
nlp.jtprod = jtprod
nlp.hiprod = hiprod

J = nlp.jac(np.array([1.,1.,1.]))
Jtest = np.zeros([1,3])
Jtest[:,0] = J * np.array([1.,0.,0.])
Jtest[:,1] = J * np.array([0.,1.,0.])
Jtest[:,2] = J * np.array([0.,0.,1.])

print 'J orig = \n', Jtest


def hessian_analytic(x):
    H = np.zeros([3,3])
    H[0,0] = 2. - x[1]*np.sin(x[0]+x[2])
    H[0,1] = np.cos(x[0]+x[2])
    H[0,2] = -x[1]*np.sin(x[0]+x[2])

    H[1,0] = np.cos(x[0]+x[2])
    H[1,1] = 36.*x[1]**2*x[2]**4
    H[1,2] = np.cos(x[0]+x[2]) + 48.*x[1]**3*x[2]**3

    H[2,0] = -x[1]*np.sin(x[0]+x[2])
    H[2,1] = np.cos(x[0]+x[2]) + 48.*x[1]**3*x[2]**3
    H[2,2] = -x[1]*np.sin(x[0]+x[2]) + 36.*x[1]**4*x[2]**3
    return H

Hop = nlp.hess(np.array([1.,1.,1.]))
Htest = np.zeros([3,3])

Htest[:,0] = Hop * np.array([1.,0.,0.])
Htest[:,1] = Hop * np.array([0.,1.,0.])
Htest[:,2] = Hop * np.array([0.,0.,1.])


print 'True Hessian - Test hessian = \n', \
             hessian_analytic(np.array([1.,1.,1.])) - Htest

#print Hop.T * np.array([1.,0.,0.])


slacknlp = SlackNLP(nlp, keep_variable_bounds=True)

print slacknlp.Lvar
print slacknlp.Lcon


J = slacknlp.jac(np.array([1.,1.,1.,0.]))

Jtest = np.zeros([1,4])
Jtest[:,0] = J * np.array([1.,0.,0.,0.])
Jtest[:,1] = J * np.array([0.,1.,0.,0.])
Jtest[:,2] = J * np.array([0.,0.,1.,0.])
Jtest[:,3] = J * np.array([0.,0.,0.,1.])
#Jtest[:,4] = J * np.array([0.,0.,0.,0.,1.])


print Jtest

Jttest = np.zeros([4,1])
Jttest[:,0] = J.T * np.array([1.,0.])
#Jttest[:,1] = J.T * np.array([0.,1.])

print Jttest



J = slacknlp.jac(np.array([1.,1.,1.,0.]))

Ltest = np.zeros([4,4])
Ltest[:,0] = J.T *(J * np.array([1.,0.,0.,0.]))
Ltest[:,1] = J.T *(J * np.array([0.,1.,0.,0.]))
Ltest[:,2] = J.T *(J * np.array([0.,0.,1.,0.]))
Ltest[:,3] = J.T *(J * np.array([0.,0.,0.,1.]))

print 'Ltest = \n', Ltest

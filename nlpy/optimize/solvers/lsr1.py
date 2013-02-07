"""
lsr1.py

A class containing a limited-memory Symmetric Rank-one (LSR1) approximation
to a symmetric matrix. This approximation may not be positive-definite and is
useful in trust region methods for optimization.
"""

# =============================================================================
# External Python modules
# =============================================================================
import numpy as np
import numpy.linalg
import logging

# =============================================================================
# LSR1 Class
# =============================================================================
class LSR1(object):
    """
    Class LSR1 is similar to LBFGS, except that it uses a different
    approximation scheme. Inheritance is currently taken through InverseLBFGS
    to avoid multiple-inheritance confusion.

    This class is useful in trust region methods, where the approximate Hessian
    is used in the model problem. LSR1 has the advantatge over LBFGS and LDFP
    of permitting approximations that are not positive-definite.
    """

    def __init__(self, n, npairs=5, **kwargs):
        # Mandatory arguments
        self.n = n
        self.npairs = npairs

        # Optional arguments
        self.scaling = kwargs.get('scaling', False)

        # insert to points to the location where the *next* (s,y) pair
        # is to be inserted in self.s and self.y.
        self.insert = 0

        # Threshold on dot product s'y to accept a new pair (s,y).
        self.accept_threshold = 1.0e-8

        # Storage of the (s,y) pairs
        self.s = np.empty((self.n, self.npairs), 'd')
        self.y = np.empty((self.n, self.npairs), 'd')
        self.alpha = np.empty(self.npairs, 'd')
        self.ys = [None] * self.npairs
        self.gamma = 1.0

        # Keep track of number of matrix-vector products.
        self.numMatVecs = 0

        logger_name = kwargs.get('logger_name', 'nlpy.lsr1')
        self.log = logging.getLogger(logger_name)
        #self.log.addHandler(logging.NullHandler())
        self.log.info('Logger created')

    def store(self, new_s, new_y):
        """
        Store the new pair (new_s,new_y). A new pair
        is only accepted if 
        | s_k' (y_k -B_k s_k) | >= 1e-8 ||s_k|| ||y_k - B_k s_k ||.
        """
        Bs = self.matvec(new_s)
        ymBs = new_y - Bs
        criterion = abs(np.dot(ymBs, new_s))

        if criterion >= self.accept_threshold * np.linalg.norm(new_s) * np.linalg.norm(ymBs) and criterion >= 1e-15 :
            insert = self.insert
            self.s[:,insert] = new_s.copy()
            self.y[:,insert] = new_y.copy()
            self.ys[insert] = np.dot(new_y, new_s)
            self.insert += 1
            self.insert = self.insert % self.npairs
        else:
            self.log.debug('Not accepting LSR1 update: |<y-Bs,s>|= %g' % criterion)
        return


    def restart(self):
        """
        Restart the approximation by clearing all data on past updates.
        """
        self.ys = [None] * self.npairs
        self.s = np.empty((self.n, self.npairs), 'd')
        self.y = np.empty((self.n, self.npairs), 'd')
        self.insert = 0
        return


    def matvec(self, v):
        """
        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using
        the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and storing key dot products.
        """
        self.numMatVecs += 1

        q = v.copy()
        s = self.s ; y = self.y ; ys = self.ys
        npairs = self.npairs
        a = np.zeros([npairs,1],'d')
        minimat = np.zeros([npairs,npairs],'d')

        if self.scaling:
            last = (self.insert - 1) % npairs
            if ys[last] is not None:
                self.gamma = ys[last]/np.dot(y[:,last],y[:,last])
                q /= self.gamma

        paircount = 0
        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                a[k] = np.dot(y[:,k],v[:]) - np.dot(s[:,k],q[:])
                paircount += 1

        # Populate small matrix to be inverted
        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                minimat[k,k] = ys[k] - np.dot(s[:,k],s[:,k])/self.gamma
                for j in range(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        minimat[k,l] = np.dot(s[:,k],y[:,l]) - np.dot(s[:,k],s[:,l])/self.gamma
                        minimat[l,k] = minimat[k,l]

        if paircount > 0:
            rng = paircount
            b = np.linalg.solve(minimat[0:rng,0:rng],a[0:rng])

        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                q += b[k]*y[:,k] - b[k]/self.gamma*s[:,k]
        return q


class LSR1_unrolling(LSR1):
    """
    LSR1 quasi newton using an unrolling formula.
    For this procedure see [Nocedal06]
    """
    def __init__(self, n, npairs=5, **kwargs):
        LSR1.__init__(self, n, npairs, **kwargs)
        self.accept_threshold = 1e-8

    def matvec(self, v):
        """
        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using
        the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and storing key dot products.
        """
        self.numMatVecs += 1

        q = v.copy()
        s = self.s ; y = self.y ; ys = self.ys
        npairs = self.npairs
        a = np.zeros([self.n, npairs])
        aTs = np.zeros([npairs,1])

        if self.scaling:
            last = (self.insert - 1) % npairs
            if ys[last] is not None:
                self.gamma = ys[last]/np.dot(y[:,last],y[:,last])
                q /= self.gamma

        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                a[:,k] = y[:,k] - s[:,k]/self.gamma
                for j in range(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        a[:,k] -= np.dot(a[:,l], s[:,k])/aTs[l] * a[:,l]
                aTs[k] = np.dot(a[:,k], s[:,k])
                q += np.dot(a[:,k],v[:])/aTs[k]*a[:,k]
        return q



class InverseLSR1(LSR1):
    """
    Class InverseLSR1 is similar to InverseLBFGS, except that it uses a different
    approximation scheme. Inheritance is currently taken through LSR1
    to avoid multiple-inheritance confusion.

    This class is useful in trust region methods, where the approximate Hessian
    is used in the model problem. LSR1 has the advantatge over LBFGS and LDFP
    of permitting approximations that are not positive-definite.
    """

    def __init__(self, n, npairs=5, **kwargs):
        LSR1.__init__(self, n, npairs, **kwargs)
        self.accept_threshold = 1e-8

    def matvec(self, v):
        """
        Compute a matrix-vector product between the current limited-memory
        approximation to the Hessian matrix and the vector v using
        the outer product representation.

        Note: there is probably some optimization that could be done in this
        function with respect to memory use and storing key dot products.
        """
        self.numMatVecs += 1

        q = v.copy()
        s = self.s ; y = self.y ; ys = self.ys
        npairs = self.npairs
        a = np.zeros(npairs,'d')
        minimat = np.zeros([npairs,npairs],'d')

        if self.scaling:
            last = (self.insert - 1) % npairs
            if ys[last] is not None:
                self.gamma = ys[last]/np.dot(y[:,last],y[:,last])
                q *= self.gamma

        paircount = 0
        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                a[k] = np.dot(s[:,k],v[:]) - np.dot(y[:,k],q[:])
                paircount += 1

        # Populate small matrix to be inverted
        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                minimat[k,k] = -ys[k] - np.dot(y[:,k],y[:,k])*self.gamma
                for j in range(i):
                    l = (self.insert + j) % npairs
                    if ys[l] is not None:
                        minimat[k,l] = np.dot(y[:,k],s[:,l]) - np.dot(y[:,k],y[:,l])*self.gamma
                        minimat[l,k] = minimat[k,l]

        if paircount > 0:
            rng = paircount
            b = np.linalg.solve(minimat[0:rng,0:rng],a[0:rng])

        for i in range(npairs):
            k = (self.insert + i) % npairs
            if ys[k] is not None:
                q += b[k]*s[:,k] - b[k]*self.gamma*y[:,k]

        return q


# end class

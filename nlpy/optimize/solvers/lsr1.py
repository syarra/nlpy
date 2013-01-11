"""
lsr1.py

A class containing a limited-memory Symmetric Rank-one (LSR1) approximation
to a symmetric matrix. This approximation may not be positive-definite and is
useful in trust region methods for optimization.
"""

# =============================================================================
# External Python modules
# =============================================================================
import numpy
import numpy.linalg

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
        self.scaling = kwargs.get('scaling', True)

        # insert to points to the location where the *next* (s,y) pair
        # is to be inserted in self.s and self.y.
        self.insert = 0

        # Threshold on dot product s'y to accept a new pair (s,y).
        self.accept_threshold = 1.0e-8

        # Storage of the (s,y) pairs
        self.s = numpy.empty((self.n, self.npairs), 'd')
        self.y = numpy.empty((self.n, self.npairs), 'd')
        self.alpha = numpy.empty(self.npairs, 'd')
        self.ys = [None] * self.npairs
        self.gamma = 1.0

        # Keep track of number of matrix-vector products.
        self.numMatVecs = 0

    def store(self, new_s, new_y):
        """
        Store the new pair (new_s,new_y). A new pair
        is only accepted if 
        | s_k' (y_k -B_k s_k) | >= 1e-8 ||s_k|| ||y_k - B_k s_k ||.
        """
        Bs = self.matvec(new_s)
        criterion = abs(numpy.dot(new_s, new_y - Bs))

        if criterion >= self.accept_threshold * numpy.linalg.norm(new_s) * numpy.linalg.norm(new_y - Bs):
            insert = self.insert
            self.s[:,insert] = new_s.copy()
            self.y[:,insert] = new_y.copy()
            self.ys[insert] = numpy.dot(new_y, new_s)
            #print 'ys', self.ys[insert]
            self.insert += 1
            self.insert = self.insert % self.npairs
        else:
            print 'Skipping this pair'
        return


    def restart(self):
        """
        Restart the approximation by clearing all data on past updates.
        """
        self.ys = [None] * self.npairs
        self.s = numpy.empty((self.n, self.npairs), 'd')
        self.y = numpy.empty((self.n, self.npairs), 'd')
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
        a = numpy.zeros(npairs,'d')
        minimat = numpy.zeros([npairs,npairs],'d')

        if self.scaling:
            last = (self.insert - 1) % self.npairs
            if ys[last] is not None:
                self.gamma = ys[last]/numpy.dot(y[:,last],y[:,last])
                q /= self.gamma

        paircount = 0
        for i in range(npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                a[paircount] = numpy.dot(y[:,k],v[:]) - numpy.dot(s[:,k],q[:])
                paircount += 1

        # Populate small matrix to be inverted
        k_ind = 0
        for i in range(npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                minimat[k_ind,k_ind] = ys[k] - numpy.dot(s[:,k],s[:,k])/self.gamma
                l_ind = 0
                for j in range(i):
                    l = (self.insert + j) % self.npairs
                    if ys[l] is not None:
                        minimat[k_ind,l_ind] = numpy.dot(y[:,k],s[:,l]) - numpy.dot(s[:,k],s[:,l])/self.gamma
                        minimat[l_ind,k_ind] = minimat[k_ind,l_ind]
                        l_ind += 1
                k_ind += 1

        #print minimat
        if paircount > 0:
            rng = paircount
            b = numpy.linalg.solve(minimat[0:rng,0:rng],a[0:rng])

        for i in range(paircount):
            k = (self.insert - paircount + i) % self.npairs
            q += numpy.dot(b[i],y[:,k]) - numpy.dot(b[i],s[:,k]/self.gamma)

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
        a = numpy.zeros(npairs,'d')
        minimat = numpy.zeros([npairs,npairs],'d')

        if self.scaling:
            last = (self.insert - 1) % self.npairs
            if ys[last] is not None:
                self.gamma = ys[last]/numpy.dot(y[:,last],y[:,last])
                q /= self.gamma

        paircount = 0
        for i in range(npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                a[paircount] = numpy.dot(s[:,k],v[:]) - numpy.dot(y[:,k],q[:])
                paircount += 1

        # Populate small matrix to be inverted
        k_ind = 0
        for i in range(npairs):
            k = (self.insert + i) % self.npairs
            if ys[k] is not None:
                minimat[k_ind,k_ind] = ys[k] - numpy.dot(y[:,k],y[:,k])/self.gamma
                l_ind = 0
                for j in range(i):
                    l = (self.insert + j) % self.npairs
                    if ys[l] is not None:
                        minimat[k_ind,l_ind] = numpy.dot(s[:,k],y[:,l]) - numpy.dot(y[:,k],y[:,l])/self.gamma
                        minimat[l_ind,k_ind] = minimat[k_ind,l_ind]
                        l_ind += 1
                k_ind += 1

        if paircount > 0:
            rng = paircount
            b = numpy.linalg.solve(minimat[0:rng,0:rng],a[0:rng])

        for i in range(paircount):
            k = (self.insert - paircount + i) % self.npairs
            q += numpy.dot(b[i], s[:,k]) - numpy.dot(b[i]/self.gamma, y[:,k])

        return q


# end class

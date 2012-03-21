'''
lsr1.py

A class containing a limited-memory Symmetric Rank-one (LSR1) approximation 
to a symmetric matrix. This approximation may not be positive-definite and is 
useful in trust region methods for optimization.
'''

# =============================================================================
# Standard Python modules
# =============================================================================
import sys

# =============================================================================
# External Python modules
# =============================================================================
import numpy
import numpy.linalg

# =============================================================================
# Extension modules
# =============================================================================
from nlpy.optimize.solvers.lbfgs import InverseLBFGS

# =============================================================================
# LSR1 Class
# =============================================================================
class LSR1(InverseLBFGS):
	"""
	Class LSR1 is similar to LBFGS, except that it uses a different 
	approximation scheme. Inheritance is currently taken through InverseLBFGS 
	to avoid multiple-inheritance confusion.

	This class is useful in trust region methods, where the approximate Hessian 
	is used in the model problem. LSR1 has the advantatge over LBFGS and LDFP 
	of permitting approximations that are not positive-definite.
	"""

	def __init__(self, n, npairs=5, **kwargs):
		InverseLBFGS.__init__(self, n, npairs, **kwargs)

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
						minimat[k_ind,l_ind] = numpy.dot(s[:,k],y[:,l]) - numpy.dot(s[:,k],s[:,l])/self.gamma
						minimat[l_ind,k_ind] = minimat[k_ind,l_ind]
						l_ind += 1
				k_ind += 1

		if paircount > 0:
			rng = paircount
			b = numpy.linalg.solve(minimat[0:rng,0:rng],a[0:rng])

		for i in range(paircount):
			k = (self.insert - paircount + i) % self.npairs
			q += b[i]*y[:,k] - (b[i]/self.gamma)*s[:,k]

		return q


# end class
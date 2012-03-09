'''
mfnlp.py

An abstract class of a matrix-free NLP model for NLPy.
'''

# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys
import pdb
import copy

# =============================================================================
# External Python modules
# =============================================================================
import numpy as np

# =============================================================================
# Extension modules
# =============================================================================
from nlpy.model.nlp import NLPModel

# =============================================================================
# Matrix-Free Problem Class
# =============================================================================
class MFModel(NLPModel):

	'''
	MFModel is a derived type of NLPModel which focuses on matrix-free 
	implementations of the standard NLP.

	Most of the functionality is the same as NLPModel except for additional 
	methods and counters for Jacobian-free cases.

	Note: these parts could be reintegrated into NLPModel at a later date.
	'''

	def __init__(self, n=0, m=0, name='Generic Matrix-Free', **kwargs):

		# Standard NLP initialization
		NLPModel.__init__(self,n=n,m=m,name=name,**kwargs)

		# Additional elements for this class
		self.JTprod = 0	# Counter for Jacobian-transpose products

	# end def

	def ResetCounters(self):

		NLPModel.ResetCounters()
		self.JTprod = 0

	# end def

    # Evaluate matrix-vector product between
    # the Jacobian and a vector
	def jprod(self, x, p, **kwargs):
		raise NotImplementedError, 'This method must be subclassed.'
	# end def

    # Evaluate matrix-vector product between
    # the transpose Jacobian and a vector
	def jtprod(self, x, q, **kwargs):
		raise NotImplementedError, 'This method must be subclassed.'
	# end def

# end class

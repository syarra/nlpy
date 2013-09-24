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
from nlpy.model.snlp import SlackNLP
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


class MFSlackNLP( SlackNLP):
    def __init__(self, nlp, **kwargs):

        # Standard initializations

        SlackNLP.__init__(self,nlp,**kwargs)

    def jac(self, x, **kwargs):
        return SimpleLinearOperator(self.n, self.m, symmetric=False,
                         matvec=lambda u: self.jprod(x,u,**kwargs),
                         matvec_transp=lambda u: self.jtprod(x,u,**kwargs))

    # Caution here ! 
    # we defined hprod in auglag to be the original size hessian product.
    # so here hprod is a on vector and not a n vector.
    # This should be fixed.

    def hprod(self, x, y, v, **kwargs):
        on = self.original_n ; om = self.original_m
        Hv = self.nlp.hprod(x[:on], y[:om], v[:on], **kwargs)
        return Hv

    def hess(self, x, z=None, **kwargs):
        return SimpleLinearOperator(self.original_n, self.original_n, symmetric=True,
                         matvec=lambda u: self.hprod(x,z,u,**kwargs))

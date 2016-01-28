"""
2015-2016 Constantine Belev const.belev@ya.ru
"""

from black_box import BlackBox
from cg_approximator import CGApproximator
from gd_approximator import GDApproximator
from mgd_approximator import MGDApproximator
from lowrank_matrix import ManifoldElement
from manifold_functions import TangentVector, riemannian_grad_full, gradient_projection_partial
from manifold_functions import riemannian_grad_partial, delta_on_sigma_set, svd_retraction



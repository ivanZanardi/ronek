import numpy as np
import scipy as sp

from .coarse_graining_m0 import CoarseGrainingM0


class CoarseGrainingM1(CoarseGrainingM0):
  """
  See:
    http://dx.doi.org/10.1063/1.4915926
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    molecule,
    T=None
  ):
    super(CoarseGrainingM1, self).__init__(molecule, T)

  # Calling
  # ===================================
  def build(
    self,
    mapping=None,
    nb_bins=1
  ):
    self.set_probmat(mapping, nb_bins)
    # Test bases
    e = self.molecule.lev["e"].reshape(-1,1)
    self.psi = np.hstack([self.P, self.P*e])

  def encode(self, x):
    return x @ self.psi

  def decode(self, x):
    # Setting up
    is_2d = (len(x.shape) > 1)
    if (is_2d and (x.shape[-1] == 2*self.nb_bins)):
      x = x.T
    pfun = self.molecule.q_int_2d if is_2d else self.molecule.q_int
    # Extract group number densities and temperatures
    ng, Tg = np.split(x, 2, axis=0)
    # Loop over groups/bins
    n = 0.0
    for g in range(self.nb_bins):
      q = (self.P[:,g] * pfun(Tg[g])).T
      Q = np.sum(q, axis=0)
      n = n + ng[g]*q/Q
    return n

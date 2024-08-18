import numpy as np
import scipy as sp


class CoarseGraining(object):
  """
  See:
    http://dx.doi.org/10.1063/1.4915926
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    molecule
  ):
    self.molecule = molecule

  # Calling
  # ===================================
  def __call__(
    self,
    mapping=None,
    nb_bins=1
  ):
    if (mapping is None):
      mapping = self.get_mapping(nb_bins)
    P = self.construct_probmat(mapping)
    # Trial bases
    phi = P * self.molecule.q.reshape(-1,1)
    Q = P.T @ self.molecule.q
    phi /= Q.reshape(1,-1)
    # Test bases
    psi = P
    return phi, psi

  def get_mapping(self, nb_bins, eps=1e-6):
    """Energy-based binning"""
    # Min/max energies
    e = self.molecule.lev["e"]
    e_min, e_max = np.amin(e), np.amax(e)
    # Dissociation energy
    e_d = min(self.molecule.lev["e_d"], e_max)
    # Number of bound and quasi-bound bins
    nb_b = round(e_d/e_max*nb_bins)
    if ((nb_b == nb_bins) and (e_d < e_max)):
      nb_b -= 1
    nb_qb = nb_bins - nb_b
    # Energy intervals
    inter_b = np.linspace(e_min, e_d, nb_b+1)
    inter_qb = np.linspace(e_d, e_max*(1.0+eps), nb_qb+1)
    intervals = np.concatenate([inter_b, inter_qb[1:]])
    # Define mapping
    lev_to_bin = (e.reshape(-1,1) >= intervals.reshape(1,-1))
    lev_to_bin = np.sum(lev_to_bin, axis=1)
    return lev_to_bin

  def construct_probmat(self, mapping):
    """Probability matrix"""
    mapping = (mapping - np.amin(mapping)).astype(int)
    nb_lev, nb_bins = self.molecule.nb_comp, np.amax(mapping)+1
    data = np.ones_like(nb_lev)
    indices = (np.arange(nb_lev), mapping)
    shape = (nb_lev, nb_bins)
    return sp.sparse.coo_matrix((data, indices), shape).toarray()

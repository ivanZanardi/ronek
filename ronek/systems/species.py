import json
import numpy as np

from .. import const


class Species(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    properties,
    use_factorial=False
  ):
    # Load properties
    if (not isinstance(properties, dict)):
      with open(properties, 'r') as file:
        properties = json.load(file)
    # Set properties
    for (k, v) in properties.items():
      setattr(self, k, v)
    self.lev = {k: np.array(v).reshape(-1) for (k, v) in self.lev.items()}
    # Number of pseudo-species
    self.nb_comp = len(self.lev["e"])
    # Thermo
    self.q = None
    # Control variables
    self.use_factorial = use_factorial

  def compute_mom(self, n, m=0):
    e = self.lev["e"] / const.eV_to_J
    return np.sum(n * e**m, axis=0)

  def compute_mom_basis(self, max_mom):
    e = self.lev["e"] / const.eV_to_J
    m = [np.ones_like(e)]
    for i in range(max_mom-1):
      mi = m[-1]*e
      if self.use_factorial:
        mi /= (i+1)
      m.append(mi)
    return np.vstack(m)

  # Partition functions
  # ===================================
  def update(self, T):
    self.q = self.q_tot(T)

  def q_tot(self, T):
    return self.q_tra(T) * self.q_che(T) * self.q_int(T)

  def q_tra(self, T):
    base = 2.0 * np.pi * self.m * const.UKB / (const.UH**2)
    return np.power(base*T, 1.5)

  def q_che(self, T):
    return np.exp(-self.e_f/(const.UKB*T))

  def q_int(self, T):
    return self.lev["g"] * np.exp(-self.lev["e"]/(const.UKB*T))

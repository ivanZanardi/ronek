import numpy as np

from silx.io.dictdump import h5todict


class Kinetics(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    rates,
    species
  ):
    # Load rates
    self.rates = rates
    if (not isinstance(self.rates, dict)):
      self.rates = h5todict(self.rates)
    # Set species
    self.species = species

  # Rates
  # ===================================
  def update(self, T):
    # Get partition functions
    q = [self.species[k].q for k in ("atom", "molecule")]
    # Compute fwd reaction rates
    rates = self.compute_fwd_rates(T)
    # Compute bwd reaction rates
    self.rates = self.compute_bwd_rates(rates, *q)

  def compute_fwd_rates(self, T):
    rates = {}
    # Loop over collisions
    for c in self.rates.keys():
      rates[c] = {}
      # Loop over processes
      for p in self.rates[c].keys():
        # Extract Arrhenius law parameters
        param = [self.rates[c][p][k] for k in ("A", "beta", "Ta")]
        # Apply Arrhenius law
        rates[c][p] = {"fwd": self.arrhenius(T, *param)}
    return rates

  def arrhenius(self, T, A, beta, Ta):
    return A * np.exp(beta*np.log(T) - Ta/T)

  def compute_bwd_rates(self, rates, q_a, q_m):
    # Loop over collisions
    for c in self.rates.keys():
      # Loop over processes
      for p in self.rates[c].keys():
        # Initialize 'bwd' rates
        i_rates = rates[c][p]["fwd"]
        # Set shapes
        r_shape = i_rates.shape
        nb_species = len(r_shape)
        q_shape = tuple([1]*nb_species)
        for i in range(nb_species):
          # Define i-th partition function
          qi = q_m if (r_shape[i] > 1) else q_a
          # > Products
          if (i > 1):
            qi = 1.0/qi
          # Reshape i-th partition function
          qi_shape = list(q_shape)
          qi_shape[i] = r_shape[i]
          qi = qi.reshape(qi_shape)
          # Apply i-th partition function
          i_rates = i_rates * qi
        # Transpose 'bwd' rates
        axes = np.arange(len(r_shape))
        i_rates = np.transpose(i_rates, axes=np.roll(axes, shift=-2))
        # Squeeze and store rates
        rates[c][p]["bwd"] = i_rates.squeeze()
        rates[c][p]["fwd"] = rates[c][p]["fwd"].squeeze()
    return rates

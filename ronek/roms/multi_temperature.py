import numpy as np
import scipy as sp
import pandas as pd

from ronek import const
from ronek.systems.species import Species


class MultiTemperature(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    molecule,
    T=None
  ):
    self.molecule = Species(molecule)
    if (T is not None):
      self.molecule.update(float(T))

  # Calling
  # ===================================
  def __call__(self, filename, teval=None):
    data = pd.read_csv(filename)
    # Moments
    # > Order 0
    n = (data[f"X_{self.molecule.name}"] * data[["n"]]).values
    # > Order 1
    Tr, Tv = [data[k].values for k in ("Th", "Tv")]
    e = self.compute_energy(Tr, Tv)
    # Interpolate
    x = np.vstack([n, e]).T
    if (teval is not None):
      t = data["t"].values
      x = sp.interpolate.interp1d(t, x, kind="cubic", axis=0)(teval)
    return x

  def compute_energy(self, Tr, Tv):
    # Rigid-rotor-harmonic oscillator model
    # Rotational energy
    er = self.molecule.R * Tr
    # Vibrational energy
    r = self.molecule.theta_v / Tv
    ev = self.molecule.R * Tv * r / (np.exp(r)-1.0)
    # Total internal energy
    e = (er + ev) * self.molecule.m / const.eV_to_J
    return e

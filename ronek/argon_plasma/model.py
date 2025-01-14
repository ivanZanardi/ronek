import abc
import time
import numpy as np
import scipy as sp
import pandas as pd

from pyDOE import lhs

from .. import const
from .. import utils
from .mixture import Mixture
from .kinetics import Kinetics
from typing import Dict, List, Optional, Tuple


class Model(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    species: Dict[str, str],
    kin_dtb: str,
    rad_dtb: Optional[str] = None,
    use_rad: bool = False,
    use_proj: bool = False,
    use_einsum: bool = False,
    use_factorial: bool = False,
    use_coll_int_fit: bool = False
  ) -> None:
    # Thermochemistry database
    # -------------
    # Mixture
    self.mix = Mixture(
      species,
      species_order=("Ar", "Arp", "em"),
      use_factorial=use_factorial
    )
    self.mix.build()
    self.nb_eqs = self.mix.nb_comp + 2
    # Kinetics
    self.kin = Kinetics(
      mixture=self.mix,
      reactions=kin_dtb,
      use_fit=use_coll_int_fit
    )
    # Radiation
    self.use_rad = use_rad
    self.rad = None
    # FOM
    # -------------
    # Solving
    self.use_einsum = use_einsum
    self.fun = None
    self.jac = None
    # ROM
    # -------------
    self.use_proj = use_proj
    # Bases
    self.phi = None
    self.psi = None
    self.runtime = 0.0

  def _is_einsum_used(self, identifier: str) -> None:
    if self.use_einsum:
      raise NotImplementedError(
        "This functionality is not supported " \
        f"when using 'einsum': '{identifier}'."
      )

  # ROM
  # ===================================
  def set_basis(
    self,
    phi: np.ndarray,
    psi: np.ndarray
  ) -> None:
    self.phi, self.psi = phi, psi
    # Biorthogonalize
    self.phi = self.phi @ sp.linalg.inv(self.psi.T @ self.phi)
    # Projector
    self.P = self.phi @ self.psi.T

  # FOM
  # ===================================
  def _fun(self, t, x, rho):
    # Extract variables
    w, e, e_e = self.extract_vars(x)
    # Update composition
    self.mix.update_composition(w, rho)
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    # Compute temperatures
    T, Te = self.mix.compute_temp(e, e_e)
    # Update thermo
    self.mix.update_species_thermo(T, Te)
    # Compose kinetic operators
    kops = self.compose_kin_ops(T, Te)
    # > Ionization processes
    iops_fwd = (kops["Ih"]["fwd"] * nn[0] + kops["Ie"]["fwd"] * ne) * nn
    iops_bwd = (kops["Ih"]["bwd"] * nn[0] + kops["Ie"]["bwd"] * ne) * ni * ne
    iops = iops_fwd.T - iops_bwd
    # Right hand side
    f = np.zeros(self.nb_eqs)
    # > Argon
    i = self.mix.species["Ar"].indices
    f[i] = (kops["EXh"] * nn[0] + kops["EXe"] * ne) @ nn \
         - np.sum(iops, axis=1)
    # > Argon Ion
    i = self.mix.species["Arp"].indices
    fion = np.sum(iops, axis=0)
    f[i] = fion
    # > Electron mass
    i = self.mix.species["em"].indices
    f[i] = np.sum(fion)
    # > Electron energy
    f[-1] = np.sum(kops["EXe_e"] @ nn) \
          - np.sum(kops["Ie_e"]["fwd"] * nn) \
          + np.sum(kops["Ie_e"]["bwd"] * ni * ne)
    f[-1] *= ne
    # Conversions
    # > Number densities to mass fractions
    f[:self.mix.nb_comp] = self.mix.get_w(f[:self.mix.nb_comp], rho)


    # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return f

  def extract_vars(self, x):
    # Mass fractions
    w = x[:self.mix.nb_comp]
    # Total and electron energies
    e, e_e = x[self.mix.nb_comp:]
    return w, e, e_e

  def compose_kin_ops(self, T, Te):
    # Update kinetics
    self.kin.update(T, Te)
    # Compose operators
    ops = {}
    # > Excitation processes
    for k in ("EXh", "EXe"):
      rates = self.kin.rates[k]
      ops[k] = self._compose_kin_ops_ex(rates)
      if (k == "EXe"):
        ops[k+"_e"] = self._compose_kin_ops_ex(rates, apply_energy=True)
    # > Ionization processes
    for k in ("Ih", "Ie"):
      rates = self.kin.rates[k]
      ops[k] = self._compose_kin_ops_ion(rates)
      if (k == "Ie"):
        ops[k+"_e"] = self._compose_kin_ops_ion(rates, apply_energy=True)
    return ops

  def _compose_kin_ops_ex(self, rates, apply_energy=False):
    k = rates["fwd"] + rates["bwd"]
    k = (k - np.diag(np.sum(k, axis=-1))).T
    if apply_energy:
      k *= self.mix.de["nn"]
    return k

  def _compose_kin_ops_ion(self, rates, apply_energy=False):
    k = {d: rates[d].T for d in ("fwd", "bwd")}
    if apply_energy:
      k["fwd"] *= self.mix.de["in"].T
      k["bwd"] *= self.mix.de["in"]
    return k





  def source_term_ie_vec(self, nn, ni, ne, kf, kb):
    iops_fwd = kf.T * nn * ne
    iops_bwd = kb.T * ni * ne * ne
    iops = iops_fwd.T - iops_bwd
    de = self.mix.de["in"]
    f = np.zeros(self.nb_eqs-1)
    # > Argon
    i = self.mix.species["Ar"].indices
    f[i] = - np.sum(iops, axis=1)
    # > Argon Ion
    i = self.mix.species["Arp"].indices
    fion = np.sum(iops, axis=0)
    f[i] = fion
    # > Electron mass
    i = self.mix.species["em"].indices
    f[i] = np.sum(fion)
    # > Electron energy
    f[-1] = np.sum(kb.T * de * ni * ne) \
          - np.sum(kf.T * de.T * nn)
    f[-1] *= ne
    return f

  def source_term_ie(self, nn, ni, ne, kf, kb):

    nb_lev = self.mix.species["Ar"].nb_comp
    ei = self.mix.species["Ar"].lev["E"]
    ej = self.mix.species["Arp"].lev["E"]

    omega = np.zeros(self.nb_eqs-1)

    for i in range(nb_lev):  # i = 0 to levels_Ar - 2

      for j in range(2): 

          # Production term
          prod_term = ne * (nn[i] * kf[i,j] - ne * ni[j] * kb[j,i])

          # Update omegai_Ar for the current level i
          omega[i] -= prod_term

          # Update omega for level Arp(j)
          omega[nb_lev+j] += prod_term

          # Update omega for em
          omega[-2] += prod_term

          # Thermal nonequilibrium adjustment
          omega[-1] -= (ej[j] - ei[i]) * prod_term

    return omega





  def source_term_exe_vec(self, nn, ne, kf, kb):
    kfb = kf + kb
    k = (kfb - np.diag(np.sum(kfb, axis=-1))).T
    # ke = (ke - np.diag(np.sum(ke, axis=-1))).T
    ke = k * self.mix.de["nn"]

    nb_lev = self.mix.species["Ar"].nb_comp
    f = np.zeros(nb_lev+1)
    f[:-1] = (k @ nn) * ne
    f[-1] = np.sum((ke @ nn) * ne)
    return f

  def source_term_exe(self, nn, ne, kf, kb):

    nb_lev = self.mix.species["Ar"].nb_comp
    e = self.mix.species["Ar"].lev["E"]

    omega = np.zeros(nb_lev+1)

    for i in range(nb_lev):  # i = 0 to levels_Ar - 2

      for j in range(i + 1, nb_lev): 

          # Production term
          prod_term = ne * (nn[i] * kf[i,j] - nn[j] * kb[j,i])

          # Update omegai_Ar for the current level i
          omega[i] -= prod_term

          # Update omega for level Ar(j)
          omega[j] += prod_term

          # Thermal nonequilibrium adjustment
          omega[-1] -= (e[j] - e[i]) * prod_term

    return omega

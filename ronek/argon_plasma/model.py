import numpy as np
import scipy as sp

from .. import const
from .mixture import Mixture
from .kinetics import Kinetics
from typing import Dict, Optional


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
    self.rad = rad_dtb
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
    w, e, e_e = self._extract_vars(x)
    # Update composition
    self.mix.update_composition(w, rho)
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    # Compute temperatures
    T, Te = self.mix.compute_temp(e, e_e)
    # Update thermo
    self.mix.update_species_thermo(T, Te)
    # Compose kinetic operators
    kops = self._compose_kin_ops(T, Te)
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
    f[-1] = self._omega_ee(kops, T, Te)
    # Conversions
    # > Number densities to mass fractions: [1/(m^3 s)] -> [1/s]
    f[:self.mix.nb_comp] = self.mix.get_w(f[:self.mix.nb_comp], rho)
    # > Volumetric to massic energy: [J/(m^3 s)] -> [J/(kg s)]
    f[-2:] /= rho
    return f

  def _extract_vars(self, x):
    # Mass fractions
    w = x[:self.mix.nb_comp]
    # Total and electron energies
    e, e_e = x[self.mix.nb_comp:]
    return w, e, e_e

  def _compose_kin_ops(self, T, Te):
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

  # Thermal nonequilibrium
  # -----------------------------------
  def _omega_ee(self, kops, T, Te):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    f = np.sum(kops["EXe_e"] @ nn) \
      - np.sum(kops["Ie_e"]["fwd"] * nn) \
      + np.sum(kops["Ie_e"]["bwd"] * ni * ne) \
      + 1.5 * const.UKB * (T-Te) * self._get_nu_eh()
    return f * ne

  def _get_nu_eh(self):
    """Electron-heavy particle relaxation frequency [1/s]"""
    s = self.mix.species
    nu = self.kin.rates["EN"] * np.sum(s["Ar"].n) / s["Ar"].m \
       + self.kin.rates["EI"] * np.sum(s["Arp"].n) / s["Arp"].m
    return const.UME * nu

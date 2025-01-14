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
  def _fun(self, t, y, rho):
    # Unpack mass fractions and temperatures
    w, T, Te = y[:-2], y[-2], y[-1]
    # Update mixture
    self.mix.update(rho, w, T, Te)
    # Kinetics
    # -------------
    kin_ops = self._compose_kin_ops(T, Te)
    # > Excitation
    omega_exc = self._compute_kin_omega_exc(kin_ops)
    # > Ionization
    omega_ion = self._compute_kin_omega_ion(kin_ops)
    # Compose RHS
    # -------------
    f = np.zeros(self.nb_eqs)
    # > Argon nd
    i = self.mix.species["Ar"].indices
    f[i] = omega_exc - np.sum(omega_ion, axis=1)
    # > Argon ion nd
    i = self.mix.species["Arp"].indices
    f[i] = np.sum(omega_ion, axis=0)
    # > Electron nd
    i = self.mix.species["em"].indices
    f[i] = np.sum(omega_ion)
    # > Convert: [1/(m^3 s)] -> [kg/(m^3 s)]
    f[:-2] *= self.mix.m
    # > Temperatures
    self._omega_temps(kin_ops, T, Te, f)
    # Return RHS
    # -------------
    return (1.0/rho) * f

  # Kinetics operators
  # -----------------------------------
  def _compose_kin_ops(self, T, Te):
    """Compose kinetic operators"""
    # Update kinetics
    self.kin.update(T, Te)
    # Compose operators
    ops = {}
    # > Excitation processes
    for k in ("EXh", "EXe"):
      rates = self.kin.rates[k]
      ops[k] = self._compose_kin_ops_exc(rates)
      if (k == "EXe"):
        ops[k+"_e"] = self._compose_kin_ops_exc(rates, apply_energy=True)
    # > Ionization processes
    for k in ("Ih", "Ie"):
      rates = self.kin.rates[k]
      ops[k] = self._compose_kin_ops_ion(rates)
      if (k == "Ie"):
        ops[k+"_e"] = self._compose_kin_ops_ion(rates, apply_energy=True)
    return ops

  def _compose_kin_ops_exc(self, rates, apply_energy=False):
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

  # Kinetics production terms
  # -----------------------------------
  def _compute_kin_omega_exc(self, kin_ops):
    nn, ne = [self.mix.species[k].n for k in ("Ar", "em")]
    return (kin_ops["EXh"] * nn[0] + kin_ops["EXe"] * ne) @ nn

  def _compute_kin_omega_ion(self, kin_ops):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    omega = {}
    for k in ("fwd", "bwd"):
      omega[k] = kin_ops["Ih"][k] * nn[0] + kin_ops["Ie"][k] * ne
      omega[k] *= nn if (k == "fwd") else (ni * ne)
    return omega["fwd"].T - omega["bwd"]

  # Temperatures
  # -----------------------------------
  def _omega_temps(self, kin_ops, T, Te, f):
    # Total energy production term
    omega_e = self._omega_energy()
    # Electron energy production term
    omega_ee = self._omega_energy_e(kin_ops, T, Te)
    # Translational temperature
    f[-2] = omega_e - (omega_ee + self.mix._e_h(f))
    f[-2] /= self.mix.cv_h
    # Electron temperature
    s = self.mix.species["em"]
    f[-1] = omega_ee - s.e * f[s.indices]
    f[-1] /= (s.w * s.cv)

  def _omega_energy(self):
    return 0.0

  def _omega_energy_e(self, kin_ops, T, Te):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    f = np.sum(kin_ops["EXe_e"] @ nn) \
      - np.sum(kin_ops["Ie_e"]["fwd"] * nn) \
      + np.sum(kin_ops["Ie_e"]["bwd"] * ni * ne) \
      + 1.5 * const.UKB * (T-Te) * self._get_nu_eh()
    return f * ne

  def _get_nu_eh(self):
    """Electron-heavy particle relaxation frequency [1/s]"""
    s = self.mix.species
    nu = self.kin.rates["EN"] * np.sum(s["Ar"].n) / s["Ar"].m \
       + self.kin.rates["EI"] * np.sum(s["Arp"].n) / s["Arp"].m
    return const.UME * nu

  # Solving
  # ===================================
  def solve(
    self,
    t: np.ndarray,
    y0: np.ndarray,
    rho: float
  ) -> np.ndarray:
    sol = sp.integrate.solve_ivp(
      fun=self._fun,
      t_span=[0.0,t[-1]],
      y0=y0,
      method="LSODA",
      t_eval=t,
      args=(rho,),
      first_step=1e-14,
      rtol=1e-6,
      atol=0.0,
      jac=None,
    )
    return sol.y

  # def pre_proc(self, y, rho):
  #   # Extract variables
  #   w, e, e_e = self._extract_vars(y)
  #   # Update composition
  #   self.mix.update_composition(w, rho)
  #   # Compute temperatures
  #   T, Te = self.mix.compute_temp(e, e_e)


  # def _extract_vars(self, y):
  #   # Mass fractions
  #   w = y[:self.mix.nb_comp]
  #   # Total and electron energies
  #   e, e_e = y[self.mix.nb_comp:]
  #   return w, e, e_e

  # def post_proc()

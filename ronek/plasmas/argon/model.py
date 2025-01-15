import numpy as np
import scipy as sp

from ... import const
from ..mixture import Mixture
from .kinetics import Kinetics
from typing import Dict, Optional


class ArgonCR(object):

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

  # Build
  # ===================================
  def build(self) -> None:
    self.mix.build()
    self._build_delta_energies()

  def _build_delta_energies(self) -> None:
    en = self.species["Ar"].lev["E"]    # [J]
    ei = self.species["Arp"].lev["E"]   # [J]
    self.de = {
      # Neutral-Neutral
      "nn": en.reshape(1,-1) - en.reshape(-1,1),
      # Ion-Neutral
      "in": ei.reshape(1,-1) - en.reshape(-1,1)
    }

  # Equilibrium composition
  # ===================================
  def compute_eq_comp(
    self,
    p: float,
    T: float
  ) -> np.ndarray:
    # Solve this system of equations:
    # -------------
    # 1) Charge neutrality:
    #    x_em = x_Arp
    # 2) Mole conservation:
    #    x_em + x_Arp + x_Ar = 1
    # 3) Detailed balance:
    #    (n_Arp*n_em)/n_Ar = (Q_Arp*Q_em)/Q_Ar
    # -------------
    # Update thermo
    self.update_species_thermo(T)
    # Compute number density
    n = p / (const.UKB*T)
    # Compute coefficient for quadratic system
    f = self.species["Arp"].Q * self.species["em"].Q
    f /= (self.species["Ar"].Q * n)
    # Solve quadratic system for 'x_em'
    a = 1.0
    b = 2.0 * f
    c = -f
    x = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    # Set molar fractions
    s = self.species["em"]
    s.x = x
    s = self.species["Arp"]
    s.x = x * s.q / s.Q
    s = self.species["Ar"]
    s.x = (1.0-2.0*x) * s.q / s.Q
    # Number densities
    n = n * np.concatenate([self.species[k].x for k in self.species_order])
    # Mass fractions and density
    rho = self.get_rho(n)
    w = self.get_w(n, rho)
    # w = np.maximum(w, const.WMIN)
    return w, rho

  # Initial composition
  # ===================================
  def get_init_composition(
    self,
    p: float,
    T: float,
    noise: bool = False,
    sigma: float = 1e-2
  ) -> np.ndarray:
    # Compute equilibrium mass fractions
    w, rho = self.compute_eq_comp(p, T)
    # Add random noise
    if noise:
      # > Electron
      s = self.species["em"]
      x_em = np.clip(s.x * self._add_norm_noise(s, sigma, use_q=False), 0, 1)
      s.x = x_em
      # > Argon Ion
      s = self.species["Arp"]
      s.x = x_em * self._add_norm_noise(s, sigma)
      # > Argon
      s = self.species["Ar"]
      s.x = (1.0-2.0*x_em) * self._add_norm_noise(s, sigma)
      # Set mass fractions
      self._M("x")
      self._set_w_s()
      # Mass densities
      w = np.concatenate([self.species[k].w for k in self.species_order])
    # Return mass densities
    return w, rho

  def _add_norm_noise(self, species, sigma, use_q=True):
    f = 1.0 + sigma * np.random.rand(species.nb_comp)
    if use_q:
      f *= species.q * f
      f /= np.sum(f)
    return f

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
    # Extract mass fractions and temperatures
    w, T, Te = self._extract_vars(y, rho)
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
    self._omega_temps(kin_ops, T, Te, f, rho)
    # Return RHS
    # -------------
    return (1.0/rho) * f

  def _extract_vars(self, y, rho):
    # Unpack mass fractions and temperatures
    w, T, pe = y[:-2], y[-2], y[-1]
    # Electron temperature
    s = self.mix.species["em"]
    Te = pe / (rho*w[s.indices]*s.R)
    Te = np.maximum(Te, const.TMIN)
    print(T, Te)
    return w, T, Te

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
      k *= self.de["nn"]
    return k

  def _compose_kin_ops_ion(self, rates, apply_energy=False):
    k = {d: rates[d].T for d in ("fwd", "bwd")}
    if apply_energy:
      k["fwd"] *= self.de["in"].T
      k["bwd"] *= self.de["in"]
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
  def _omega_temps(self, kin_ops, T, Te, f, rho):
    # Total energy production term
    omega_e = self._omega_energy()
    # Electron energy production term
    omega_ee = self._omega_energy_e(kin_ops, T, Te)
    # Translational temperature
    f[-2] = omega_e - (omega_ee + self.mix._e_h(f))
    f[-2] /= self.mix.cv_h
    # # Electron temperature
    # s = self.mix.species["em"]
    # f[-1] = omega_ee - s.e * f[s.indices]
    # print(f[-1])
    # f[-1] /= (s.w * s.cv)
    # Electron pressure
    f[-1] = rho * (self.mix.species["em"].gamma - 1) * omega_ee

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
      atol=1e-20,
      jac=None,
    )
    y = sol.y
    y[-1] = self._pe_to_Te(y, rho)
    return y

  def _pe_to_Te(self, y, rho):
    s = self.mix.species["em"]
    Te = y[-1] / (rho*y[s.indices]*s.R)
    return np.maximum(Te, const.TMIN)

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

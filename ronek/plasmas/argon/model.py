import torch
import numpy as np
import scipy as sp

from ... import const
from ... import backend as bkd
from ..mixture import Mixture
from .kinetics import Kinetics


class ArgonCR(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    kin_dtb,
    rad_dtb=None,
    use_rad=False,
    use_proj=False,
    use_factorial=False,
    use_coll_int_fit=False
  ):
    # Thermochemistry database
    # -------------
    # Mixture
    self.species_order = ("Ar", "Arp", "em")
    self.mix = Mixture(
      species,
      species_order=self.species_order,
      use_factorial=use_factorial
    )
    self.mix.build()
    # Kinetics
    self.kin = Kinetics(
      mixture=self.mix,
      reactions=kin_dtb,
      use_fit=use_coll_int_fit
    )
    # Radiation
    self.use_rad = bool(use_rad)
    self.rad = rad_dtb
    # Dimensions
    self.nb_comp = self.mix.nb_comp
    self.nb_temp = 2
    self.nb_eqs = self.nb_comp + self.nb_temp
    # FOM
    # -------------
    # Jacobian
    self._jac = torch.func.jacrev(self._fun)
    # ROM
    # -------------
    # Bases
    self.phi = None
    self.psi = None
    # Projector
    self.use_proj = bool(use_proj)
    self.P = None

  # Equilibrium composition
  # ===================================
  def compute_eq_comp(self, p, T):
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
    self.mix.update_species_thermo(T)
    # Compute number density
    n = p / (const.UKB*T)
    # Compute coefficient for quadratic system
    f = self.mix.species["Arp"].Q * self.mix.species["em"].Q
    f /= (self.mix.species["Ar"].Q * n)
    # Solve quadratic system for 'x_em'
    a = 1.0
    b = 2.0 * f
    c = -f
    x = (-b+torch.sqrt(b**2-4*a*c))/(2*a)
    x = torch.clip(x, const.XMIN, 1.0)
    # Set molar fractions
    s = self.mix.species["em"]
    s.x = x
    s = self.mix.species["Arp"]
    s.x = x * s.q / s.Q
    s = self.mix.species["Ar"]
    s.x = (1.0-2.0*x) * s.q / s.Q
    # Number densities
    n = n * torch.cat([self.mix.species[k].x for k in self.species_order])
    # Density
    rho = self.mix.get_rho(n)
    return n, rho

  # Initial composition
  # ===================================
  def get_init_composition(self, p, T, noise=False, sigma=1e-2):
    # Compute equilibrium mass fractions
    n, rho = self.compute_eq_comp(p, T)
    # Add random noise
    if noise:
      # > Electron
      s = self.mix.species["em"]
      x = s.x * self._add_norm_noise(s, sigma, use_q=False)
      x = torch.clip(x, const.XMIN, 1.0)
      s.x = x
      # > Argon Ion
      s = self.mix.species["Arp"]
      s.x = x * self._add_norm_noise(s, sigma)
      # > Argon
      s = self.mix.species["Ar"]
      s.x = (1.0-2.0*x) * self._add_norm_noise(s, sigma)
      # Number densities
      n = torch.sum(n)
      n = n * torch.cat([self.mix.species[k].x for k in self.species_order])
    return n, rho

  def _add_norm_noise(self, species, sigma, use_q=True):
    f = 1.0 + sigma * torch.rand(species.nb_comp)
    if use_q:
      f *= species.q * f
      f /= torch.sum(f)
    return f

  # ROM
  # ===================================
  def set_basis(self, phi, psi):
    self.phi, self.psi = phi, psi
    # Biorthogonalize
    self.phi = self.phi @ torch.linalg.inv(self.psi.T @ self.phi)
    # Projector
    self.P = self.phi @ self.psi.T

  # FOM
  # ===================================
  def fun(self, t, y):
    f = self._fun(bkd.to_torch(y))
    return bkd.to_numpy(f)

  def jac(self, t, y):
    j = self._jac(bkd.to_torch(y))
    if (j.isnan().any() or j.isinf().any()):
      # Finite difference Jacobian
      return sp.optimize.approx_fprime(
        xk=y,
        f=lambda y: self.fun(0.0, y),
        epsilon=bkd.epsilon()
      )
    else:
      return bkd.to_numpy(j)

  def _fun(self, y):
    # Extract number densities and temperatures
    n, T, Te = self._get_prim(y)
    # Update mixture
    self.mix.update(n, T, Te)
    # Kinetics
    # -------------
    # > Operators
    kin_ops = self._compose_kin_ops(T, Te)
    # > Production terms
    omega_exc = self._compute_kin_omega_exc(kin_ops)
    omega_ion = self._compute_kin_omega_ion(kin_ops)
    # Compose RHS - Number densities
    # -------------
    # > Argon nd
    f_nn = omega_exc - torch.sum(omega_ion, dim=1)
    # > Argon ion nd
    f_ni = torch.sum(omega_ion, dim=0)
    # > Electron nd
    f_ne = torch.sum(omega_ion).reshape(1)
    # > Concatenate
    f_n = torch.cat([f_nn, f_ni, f_ne])
    # Compose RHS - Energies
    # -------------
    f_e, f_ee = self._omega_energies(T, Te, kin_ops, f_n)
    # > Concatenate
    f = torch.cat([f_n/const.UNA, f_e, f_ee])
    return f

  def _get_prim(self, y):
    # Unpacking:
    # - c  [mol/m^3] : Molar concentrations
    # - T  [K]       : Translational temperature
    # - pe [Pa]      : Electron pressure
    c, T, pe = y[:-2], y[-2], y[-1]
    # Get number densities
    n = c * const.UNA
    # Get electron temperature
    Te = self.mix.get_Te(pe, ne=n[-1])
    # Clip temperatures
    T, Te = [self._clip_temp(z) for z in (T, Te)]
    return n, T, Te

  def _clip_temp(self, T):
    return torch.clip(T, const.TMIN, const.TMAX)

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
    k = (k - torch.diag(torch.sum(k, dim=-1))).T
    if apply_energy:
      k = k * self.mix.de["Ar-Ar"]
    return k

  def _compose_kin_ops_ion(self, rates, apply_energy=False):
    k = {d: rates[d].T for d in ("fwd", "bwd")}
    if apply_energy:
      k["fwd"] = k["fwd"] * self.mix.de["Arp-Ar"].T
      k["bwd"] = k["bwd"] * self.mix.de["Arp-Ar"]
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

  # Energies
  # -----------------------------------
  def _omega_energies(self, T, Te, kin_ops, f_n):
    # Convert mass source term: [1/(m^3 s)] -> [kg/(m^3 s)]
    f_rho = self.mix.get_rho(f_n)
    # Total energy production term
    omega_e = self._omega_energy()
    # Electron energy production term
    omega_ee = self._omega_energy_e(T, Te, kin_ops)
    # Translational temperature
    f_e = omega_e - (omega_ee + self.mix._e_h(f_rho))
    f_e = f_e / (self.mix.rho * self.mix.cv_h)
    f_e = f_e.reshape(1)
    # Electron pressure
    gamma = self.mix.species["em"].gamma
    f_ee = (gamma - 1.0) * omega_ee
    f_ee = f_ee.reshape(1)
    return f_e, f_ee

  def _omega_energy(self):
    return 0.0

  def _omega_energy_e(self, T, Te, kin_ops):
    nn, ni, ne = [self.mix.species[k].n for k in ("Ar", "Arp", "em")]
    f = torch.sum(kin_ops["EXe_e"] @ nn) \
      - torch.sum(kin_ops["Ie_e"]["fwd"] * nn) \
      + torch.sum(kin_ops["Ie_e"]["bwd"] * ni * ne) \
      + 1.5 * const.UKB * (T-Te) * self._get_nu_eh()
    return f * ne

  def _get_nu_eh(self):
    """Electron-heavy particle relaxation frequency [1/s]"""
    s = self.mix.species
    nu = self.kin.rates["EN"] * torch.sum(s["Ar"].n) / s["Ar"].m \
       + self.kin.rates["EI"] * torch.sum(s["Arp"].n) / s["Arp"].m
    return const.UME * nu

  # Solving
  # ===================================
  def solve(
    self,
    t: np.ndarray,
    y0: np.ndarray
  ) -> np.ndarray:
    self.pre_proc(y0)
    y = sp.integrate.solve_ivp(
      fun=self.fun,
      t_span=[0.0,t[-1]],
      y0=y0,
      method="BDF",
      t_eval=t,
      first_step=1e-14,
      rtol=1e-8,
      atol=0.0,
      jac=self.jac
    ).y
    self.post_proc(y)
    return y

  def pre_proc(self, y):
    # Unpacking
    n, T, Te = y[:-2], y[-2], y[-1]
    # Mixture density
    self.mix._rho(bkd.to_torch(n))
    # Electron pressure
    y[-1] = self.mix.get_pe(Te=Te, ne=n[-1])
    # Molar concentrations
    y[:-2] /= const.UNA

  def post_proc(self, y):
    # Number densities
    y[:-2] *= const.UNA
    # Electron temperature
    y[-1] = self.mix.get_Te(pe=y[-1], ne=y[-3])

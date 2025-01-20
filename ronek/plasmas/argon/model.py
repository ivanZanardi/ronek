import torch
import numpy as np
import scipy as sp

from ... import const
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
    use_pe=True,
    use_rad=False,
    use_proj=False,
    use_factorial=False,
    use_coll_int_fit=False
  ):
    # Thermochemistry database
    # -------------
    # Solve for electron pressure
    self.use_pe = bool(use_pe)
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
    # Solving
    self.fun = None
    self.jac = None
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
    # Mass fractions and density
    rho = self.mix.get_rho(n)
    w = self.mix.get_w(rho, n)
    return w, rho

  # Initial composition
  # ===================================
  def get_init_composition(self, p, T, noise=False, sigma=1e-2):
    # Compute equilibrium mass fractions
    w, rho = self.compute_eq_comp(p, T)
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
      # Set mass fractions
      self.mix._M("x")
      self.mix._set_ws()
      # Mass densities
      w = torch.cat([self.mix.species[k].w for k in self.species_order])
    # Return mass densities
    return w, rho

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
  def _fun(self, t, y, rho):
    # Extract mass fractions and temperatures
    w, T, Te = self._get_prim(y, rho)
    # Update mixture
    self.mix.update(rho, w, T, Te)
    # Kinetics
    # -------------
    # > Operators
    kin_ops = self._compose_kin_ops(T, Te)
    # > Production terms
    omega_exc = self._compute_kin_omega_exc(kin_ops)
    omega_ion = self._compute_kin_omega_ion(kin_ops)
    # Compose RHS - Mass
    # -------------
    f = torch.zeros(self.nb_eqs)
    # > Argon nd
    i = self.mix.species["Ar"].indices
    f[i] = omega_exc - torch.sum(omega_ion, dim=1)
    # > Argon ion nd
    i = self.mix.species["Arp"].indices
    f[i] = torch.sum(omega_ion, dim=0)
    # > Electron nd
    i = self.mix.species["em"].indices
    f[i] = torch.sum(omega_ion)
    # > Convert: [1/(m^3 s)] -> [1/s]
    f[:-2] = self.mix.get_w(rho, f[:-2])
    # Compose RHS - Energy
    # -------------
    self._omega_energies(rho, T, Te, kin_ops, f)
    return f

  def _get_prim(self, y, rho):
    if self.use_pe:
      w, T, pe = y[:-2], y[-2], y[-1]
      Te = self.mix.get_Te(rho, pe, w)
    else:
      w, T, Te = y[:-2], y[-2], y[-1]
    T, Te = [self._clip_temp(z) for z in (T, Te)]
    return w, T, Te

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
  def _omega_energies(self, rho, T, Te, kin_ops, f):
    # Total energy production term
    omega_e = self._omega_energy()
    # Electron energy production term
    omega_ee = self._omega_energy_e(T, Te, kin_ops)
    # Translational temperature
    f[-2] = omega_e - (omega_ee + rho*self.mix._e_h(f))
    f[-2] = f[-2] / (rho * self.mix.cv_h)
    # Electron pressure/temperature
    s = self.mix.species["em"]
    if self.use_pe:
      # > Pressure
      # See: Eq. (2.52) - Kapper's PhD thesis, The Ohio State University, 2009
      f[-1] = (s.gamma - 1.0) * omega_ee
    else:
      # > Temperature
      f[-1] = omega_ee - s.e * rho * f[s.indices]
      f[-1] = f[-1] / (s.w * rho * s.cv)

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
    y0: np.ndarray,
    rho: float
  ) -> np.ndarray:
    self.pre_proc(y0, rho)
    y = sp.integrate.solve_ivp(
      fun=self._fun,
      t_span=[0.0,t[-1]],
      y0=y0,
      method="LSODA",
      t_eval=t,
      args=(rho,),
      first_step=1e-14,
      rtol=1e-8,
      atol=0.0,
      jac=None,
    ).y
    self.post_proc(y, rho)
    return y

  def pre_proc(self, y, rho):
    if self.use_pe:
      y[-1] = self.mix.get_pe(rho=rho, Te=y[-1], w=y[:-2])

  def post_proc(self, y, rho):
    if self.use_pe:
      y[-1] = self.mix.get_Te(rho=rho, pe=y[-1], w=y[:-2])

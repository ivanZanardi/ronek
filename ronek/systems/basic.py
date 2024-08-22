import abc
import numpy as np
import scipy as sp

from .. import const
from .species import Species
from .kinetics import Kinetics


class Basic(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    rates,
    species,
    use_einsum=False,
    use_factorial=False
  ):
    # Thermochemistry database
    # -------------
    # > Species
    self.nb_eqs = 0
    self.species = {}
    for k in ("atom", "molecule"):
      if (k not in species):
        raise ValueError(
          "The 'species' input parameter should be a " \
          "dictionary with 'atom' and 'molecule' as keys."
        )
      self.species[k] = Species(species[k], use_factorial)
      self.nb_eqs += self.species[k].nb_comp
    # > Kinetics
    self.kinetics = Kinetics(rates, self.species)
    # Operators
    # -------------
    self.fom_ops = None
    self.rom_ops = None
    # Bases
    self.phi = None
    self.psi = None
    self.phif = None
    # Mass conservation operator
    self.mass_ratio = np.full((1,self.nb_eqs-1), 2.0)
    # Equilibrium ratio
    self.gamma = None
    # Solving
    # -------------
    self.use_einsum = use_einsum
    self.fom_fun = None
    self.fom_jac = None
    self.rom_fun = None
    self.rom_jac = None

  def set_eq_ratio(self, T):
    q_a, q_m = [self.species[k].q_tot(T) for k in ("atom", "molecule")]
    self.gamma = q_m / q_a**2

  def check_eq_ratio(self):
    if (self.gamma is None):
      raise ValueError("Set equilibrium ratio.")

  def check_fom_ops(self):
    if (self.fom_ops is None):
      raise ValueError("Update FOM operators.")

  def check_rom_ops(self):
    if (self.rom_ops is None):
      raise ValueError("Update ROM operators.")

  def check_basis(self):
    if (self.phi is None):
      raise ValueError("Set trial and test bases.")

  def is_einsum_used(self, identifier):
    if self.use_einsum:
      raise NotImplementedError(
        "This functionality is not supported " \
        f"when using 'einsum': '{identifier}'."
      )

  # Operators
  # ===================================
  # ROM
  # -----------------------------------
  def update_rom_ops(self, phi, psi):
    self.is_einsum_used("update_rom_ops")
    self.check_eq_ratio()
    self.check_fom_ops()
    # Set basis
    self.set_basis(phi, psi)
    # Compose operators
    self.rom_ops = self._update_rom_ops()
    self.rom_ops["m_ratio"] = self.mass_ratio @ self.phif

  def set_basis(self, phi, psi):
    self.phi, self.psi = phi, psi
    # Biorthogonalize
    self.phif = self.phi @ sp.linalg.inv(self.psi.T @ self.phi)
    # Check if complex
    for k in ("phi", "psi", "phif"):
      bases = getattr(self, k)
      if np.iscomplexobj(bases):
        setattr(self, k, bases.real)

  @abc.abstractmethod
  def _update_rom_ops(self):
    pass

  # FOM
  # -----------------------------------
  def update_fom_ops(self, T):
    # Update species
    for sp in self.species.values():
      sp.update(T)
    # Update kinetics
    self.kinetics.update(T)
    # Compose operators
    if self.use_einsum:
      self.fom_ops = self.kinetics.rates
    else:
      self.fom_ops = self._update_fom_ops(self.kinetics.rates)
    self.fom_ops["m_ratio"] = self.mass_ratio

  @abc.abstractmethod
  def _update_fom_ops(self, rates):
    pass

  # Linearized FOM
  # -----------------------------------
  def compute_lin_fom_ops(self, *args, **kwargs):
    self.is_einsum_used("compute_lin_fom_ops")
    self.check_eq_ratio()
    self.check_fom_ops()
    return self._compute_lin_fom_ops(*args, **kwargs)

  @abc.abstractmethod
  def _compute_lin_fom_ops(self, *args, **kwargs):
    pass

  # Equilibrium composition
  # -----------------------------------
  def compute_rho(self, n):
    n_a, n_m = n[:1], n[1:]
    rho_a = n_a * self.species["atom"].m
    rho_m = np.sum(n_m) * self.species["molecule"].m
    return rho_a + rho_m

  def compute_eq_comp(self, rho):
    # Solve this system of equations:
    # 1) rho_a + sum(rho_m) = rho
    # 2) n_m = gamma * n_a^2
    a = np.sum(self.gamma) * self.species["molecule"].m
    b = self.species["atom"].m
    c = -rho
    n_a = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    n_m = self.gamma*n_a**2
    return n_a, n_m

  def compute_eq_dist(self, Tint):
    q = [self.species["molecule"].q_int(Ti) for Ti in Tint]
    return [qi/np.sum(qi) for qi in q]

  # Solving
  # ===================================
  def get_init_sol(self, T, p, x_a):
    n = p / (const.UKB * T)
    n_a = np.array([n * x_a]).reshape(-1)
    q_m = self.species["molecule"].q_int(T)
    n_m = n * (1-x_a) * q_m / np.sum(q_m)
    return np.concatenate([n_a, n_m])

  def get_tgrid(self, t_lim, num):
    t = np.geomspace(*t_lim, num=num-1)
    t = np.insert(t, 0, 0.0)
    return t

  def solve(
    self,
    t,
    y0,
    fun,
    jac=None,
    ops=None,
    rtol=1e-5,
    atol=0.0,
    first_step=1e-14
  ):
    if (ops is None):
      raise ValueError("Provide set of operators as input.")
    sol = sp.integrate.solve_ivp(
      fun=fun,
      t_span=[t[0],t[-1]],
      y0=y0/const.UNA,
      method="LSODA",
      t_eval=t,
      args=(ops,),
      first_step=first_step,
      rtol=rtol,
      atol=atol,
      jac=jac
    )
    return sol.y * const.UNA

  def solve_fom(
    self,
    t,
    n0,
    rtol=1e-5,
    atol=0.0,
    first_step=1e-14
  ):
    """Solve state-to-state FOM."""
    self.check_fom_ops()
    n = self.solve(
      t=t,
      y0=n0,
      fun=self.fom_fun,
      jac=self.fom_jac,
      ops=self.fom_ops,
      rtol=rtol,
      atol=atol,
      first_step=first_step
    )
    return n[:1], n[1:]

  def solve_rom_cg(
    self,
    t,
    n0,
    rtol=1e-5,
    atol=0.0,
    first_step=1e-14
  ):
    """Solve coarse-graining-based ROM."""
    self.check_rom_ops()
    self.is_einsum_used("solve_rom_cg")
    # Encode initial condition
    z_m = self.encode(n0[1:])
    z0 = np.concatenate([n0[:1], z_m])
    # Solve
    z = self.solve(
      t=t,
      y0=z0,
      fun=self.fom_fun,
      jac=self.fom_jac,
      ops=self.rom_ops,
      rtol=rtol,
      atol=atol,
      first_step=first_step
    )
    # Decode solution
    n_m = self.decode(z[1:].T).T
    return z[:1], n_m

  def solve_rom_bt(
    self,
    t,
    n0,
    rtol=1e-5,
    atol=0.0,
    first_step=1e-14,
    use_abs=False
  ):
    """Solve balanced truncation-based ROM."""
    self.check_rom_ops()
    self.is_einsum_used("solve_rom_bt")
    # Compute equilibrium value
    rho = self.compute_rho(n=n0)
    n_a_eq, n_m_eq = self.compute_eq_comp(rho)
    self.rom_ops["n_a_eq"] = n_a_eq
    # Encode initial condition
    z_m = self.encode(n0[1:], x_eq=n_m_eq)
    z0 = np.concatenate([n0[:1], z_m])
    # Solve
    z = self.solve(
      t=t,
      y0=z0,
      fun=self.rom_fun,
      jac=self.rom_jac,
      ops=self.rom_ops,
      rtol=rtol,
      atol=atol,
      first_step=first_step
    )
    # Decode solution
    n_m = self.decode(z[1:].T, x_eq=n_m_eq).T
    if use_abs:
      n_m = np.abs(n_m)
    return z[:1], n_m

  def encode(self, x, x_eq=None):
    if (x_eq is not None):
      x = x - x_eq
    return x @ self.psi

  def decode(self, z, x_eq=None):
    x = z @ self.phif.T
    if (x_eq is not None):
      x = x + x_eq
    return x

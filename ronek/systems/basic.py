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
  def set_basis(self, phi, psi):
    self.phi, self.psi = phi, psi
    # Biorthogonalize
    self.phif = self.phi @ sp.linalg.inv(self.psi.T @ self.phi)
    # Check if complex
    for k in ("phi", "psi", "phif"):
      bases = getattr(self, k)
      if np.iscomplexobj(bases):
        setattr(self, k, bases.real)

  def update_rom_ops(self):
    self.is_einsum_used("update_rom_ops")
    self.check_eq_ratio()
    self.check_fom_ops()
    self.check_basis()
    # Compose operators
    self.rom_ops = self._update_rom_ops()
    self.rom_ops["m_ratio"] = self.mass_ratio @ self.phif

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
  def _compute_eq_comp(self, rho):
    # Solve this system of equations:
    # 1) rho_a + sum(rho_m) = rho
    # 2) n_m = gamma * n_a^2
    a = np.sum(self.gamma) * self.species["molecule"].m
    b = self.species["atom"].m
    c = -rho
    n_a = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    n_m = self.gamma*n_a**2
    return n_a, n_m

  def _compute_boltz(self, Tint):
    q = [self.species["molecule"].q_int(Ti) for Ti in Tint]
    return [qi/np.sum(qi) for qi in q]

  def _compute_rho(self, n):
    n_a, n_m = n[:1], n[1:]
    rho_a = n_a * self.species["atom"].m
    rho_m = np.sum(n_m) * self.species["molecule"].m
    return rho_a + rho_m

  # Solving
  # ===================================
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
    """
    Solve the isothermal master equation given the initial distribution
    and the translational temperature.

    Args:
      - t: A 1D vector representing the time grid (preferably distributed
           logarithmically).
      - y0: A 1D vector representing the initial solution.
      - ops: A list of operators needed to solve the system.

    Returns:
      - y: The solution at each time instant.
    """
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
    y0,
    rtol=1e-5,
    atol=0.0,
    first_step=1e-14
  ):
    self.check_fom_ops()
    return self.solve(
      t=t,
      y0=y0,
      fun=self.fom_fun,
      jac=self.fom_jac,
      ops=self.fom_ops,
      rtol=rtol,
      atol=atol,
      first_step=first_step
    )

  def solve_rom(
    self,
    t,
    y0,
    rtol=1e-5,
    atol=0.0,
    first_step=1e-14,
    use_abs=False
  ):
    self.check_rom_ops()
    # Compute equilibrium value
    rho = self._compute_rho(n=y0)
    n_a_eq, n_m_eq = self._compute_eq_comp(rho)
    self.rom_ops["n_a_eq"] = n_a_eq
    # Encode initial condition
    z = self.encode(y0[1:], x_eq=n_m_eq)
    y0 = np.concatenate([y0[:1], z])
    # Solve
    y = self.solve(
      t=t,
      y0=y0,
      fun=self.rom_fun,
      jac=self.rom_jac,
      ops=self.rom_ops,
      rtol=rtol,
      atol=atol,
      first_step=first_step
    )
    # Decode solution
    n_m = self.decode(y[1:].T, x_eq=n_m_eq).T
    n = np.vstack([y[:1], n_m])
    if use_abs:
      n = np.abs(n)
    return n

  def encode(self, x, x_eq=None):
    if (x_eq is not None):
      x = x - x_eq
    return x @ self.psi

  def decode(self, z, x_eq=None):
    x = z @ self.phif.T
    if (x_eq is not None):
      x = x + x_eq
    return x

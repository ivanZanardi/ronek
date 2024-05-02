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
    use_einsum=False
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
      self.species[k] = Species(species[k])
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
    # Solving
    # -------------
    self.use_einsum = use_einsum
    self.fun = None
    self.jac = None

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
    if self.use_einsum:
      raise NotImplementedError(
        "The implementation for constructing the ROM operators " \
        "is not available when using 'einsum' to build the RHS."
      )
    if (self.fom_ops is None):
      raise ValueError("Update FOM operators.")
    if (self.phi is None):
      raise ValueError("Set trial and test bases.")
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
  @abc.abstractmethod
  def compute_lin_fom_ops(self, *args, **kwargs):
    if self.use_einsum:
      raise NotImplementedError(
        "The implementation for constructing the linearized operators " \
        "is not available when using 'einsum' to build the RHS."
      )
    if (self.fom_ops is None):
      raise ValueError("Update FOM operators.")
    return self._compute_lin_fom_ops(*args, **kwargs)

  @abc.abstractmethod
  def _compute_lin_fom_ops(self, *args, **kwargs):
    pass

  # Equilibrium composition
  # -----------------------------------
  def _compute_eq_comp(self, p, T):
    # Solve: n_a + sum(ni) = p/(kT)
    alpha = self._compute_eq_ratio(T)
    a, n = np.sum(alpha), p/(const.UKB*T)
    n_a = (-1+np.sqrt(1+4*a*n))/(2*a)
    n_m = alpha*(n_a**2)
    return n_a, n_m

  def _compute_eq_ratio(self, T):
    q_a, q_m = [self.species[k].q_tot(T) for k in ("atom", "molecule")]
    return q_m / q_a**2

  def _compute_boltz(self, Tint):
    q = [self.species["molecule"].q_int(Ti) for Ti in Tint]
    return [qi/np.sum(qi) for qi in q]

  # Solving
  # ===================================
  def solve(
    self,
    t,
    y0,
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
      fun=self.fun,
      t_span=[t[0],t[-1]],
      y0=y0/const.UNA,
      method="LSODA",
      t_eval=t,
      args=(ops,),
      first_step=first_step,
      rtol=rtol,
      atol=atol,
      jac=self.jac
    )
    return sol.y * const.UNA

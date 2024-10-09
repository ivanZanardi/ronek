import abc
import numpy as np
import scipy as sp

from pyDOE import lhs

from .. import const
from .. import utils
from .species import Species
from .kinetics import Kinetics
from typing import Dict


class Basic(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    T,
    rates,
    species,
    use_einsum=False,
    use_factorial=False
  ):
    # Thermochemistry database
    # -------------
    self.T = float(T)
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
    self.set_eq_ratio()
    # > Kinetics
    self.kinetics = Kinetics(rates, self.species)
    # FOM
    # -------------
    # Solving
    self.use_einsum = use_einsum
    self.fun = None
    self.jac = None
    # Operators
    self.update_fom_ops()
    # ROM
    # -------------
    self.rom_valid = True
    self.rom_ops = None
    # Bases
    self.phi = None
    self.psi = None
    self.phif = None

  def set_eq_ratio(self):
    q_a, q_m = [self.species[k].q_tot(self.T) for k in ("atom", "molecule")]
    self.gamma = q_m / q_a**2

  def check_rom_ops(self):
    if (self.rom_ops is None):
      raise ValueError("Update ROM operators.")

  def is_einsum_used(self, identifier):
    if self.use_einsum:
      raise NotImplementedError(
        "This functionality is not supported " \
        f"when using 'einsum': '{identifier}'."
      )

  # Operators
  # ===================================
  # FOM
  # -----------------------------------
  def update_fom_ops(self):
    # Update species
    for sp in self.species.values():
      sp.update(self.T)
    # Update kinetics
    self.kinetics.update(self.T)
    # Compose operators
    if self.use_einsum:
      self.fom_ops = self.kinetics.rates
    else:
      self.fom_ops = self._update_fom_ops(self.kinetics.rates)
    self.mass_ratio = self.species["molecule"].m / self.species["atom"].m
    self.mass_ratio = np.full(self.nb_eqs-1, self.mass_ratio).reshape(1,-1)
    self.fom_ops["m_ratio"] = self.mass_ratio

  @abc.abstractmethod
  def _update_fom_ops(self, rates):
    pass

  # Linearized FOM
  # -----------------------------------
  def compute_lin_fom_ops(
    self,
    mu: np.ndarray,
    rho: float,
    max_mom: int = 10
  ) -> Dict[str, np.ndarray]:
    self.is_einsum_used("compute_lin_fom_ops")
    return self._compute_lin_fom_ops(mu, rho, max_mom)

  @abc.abstractmethod
  def _compute_lin_fom_ops(
    self,
    mu: np.ndarray,
    rho: float,
    max_mom: int = 10
  ) -> Dict[str, np.ndarray]:
    pass

  # ROM
  # -----------------------------------
  def update_rom_ops(self, phi, psi, biortho=True):
    self.is_einsum_used("update_rom_ops")
    # Set basis
    self.set_basis(phi, psi, biortho)
    # Compose operators
    self.rom_ops = self._update_rom_ops()
    self.rom_ops["m_ratio"] = self.mass_ratio @ self.phif

  def set_basis(self, phi, psi, biortho=True):
    self.phi, self.phif, self.psi = phi, phi, psi
    # Biorthogonalize
    if biortho:
      self.phif = self.phi @ sp.linalg.inv(self.psi.T @ self.phi)
    # Check if complex
    for k in ("phi", "psi", "phif"):
      bases = getattr(self, k)
      if np.iscomplexobj(bases):
        setattr(self, k, bases.real)
    # Check invertibility
    eps = 1e-5
    ones = np.diag(self.psi.T @ self.phi)
    if (ones < 1-eps).any():
      self.rom_valid = False
    else:
      self.rom_valid = True

  @abc.abstractmethod
  def _update_rom_ops(self):
    pass

  # Equilibrium composition
  # -----------------------------------
  def compute_rho(self, n):
    n_a, n_m = n[:1], n[1:]
    rho = n_a * self.species["atom"].m \
        + np.sum(n_m) * self.species["molecule"].m
    return rho

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

  # Solving
  # ===================================
  def get_init_sol(self, T=None, p=None, X_a=None, rho=None):
    if (p is None):
      return self._get_init_sol_from_rho(T, X_a, rho)
    else:
      return self._get_init_sol_from_p(T, X_a, p)

  def _get_init_sol_from_rho(self, T, X_a, rho, noise=False, eps=1e-2):
    # Compute mass fractions
    # > Atom
    if noise:
      X_a = np.clip(X_a + eps * np.random.rand(1), 0, 1)
    norm = X_a * self.species["atom"].m \
         + (1-X_a) * self.species["molecule"].m
    Y_a = X_a * self.species["atom"].m / norm
    # > Molecule
    q_m = self.species["molecule"].q_int(T)
    if noise:
      q_m *= (1 + eps * np.random.rand(*q_m.shape))
    Y_m = (1-Y_a) * (q_m / np.sum(q_m))
    # Compute number densities
    # > Atom
    n_a = rho * Y_a / self.species["atom"].m
    n_a = np.array(n_a).reshape(-1)
    # > Molecule
    n_m = rho * Y_m / self.species["molecule"].m
    return np.concatenate([n_a, n_m])

  def _get_init_sol_from_p(self, T, X_a, p):
    n = p / (const.UKB * T)
    n_a = np.array([n * X_a]).reshape(-1)
    q_m = self.species["molecule"].q_int(T)
    n_m = n * (1-X_a) * q_m / np.sum(q_m)
    return np.concatenate([n_a, n_m])

  def get_tgrid(self, start, stop, num):
    t = np.geomspace(start, stop, num=num-1)
    t = np.insert(t, 0, 0.0)
    return t

  def solve(
    self,
    t,
    n0,
    ops=None,
    rtol=1e-6
  ):
    if (ops is None):
      raise ValueError("Provide set of operators as input.")
    sol = sp.integrate.solve_ivp(
      fun=self.fun,
      t_span=[0.0,t[-1]],
      y0=n0/const.UNA,
      method="LSODA",
      t_eval=t,
      args=(ops,),
      first_step=1e-14,
      rtol=rtol,
      atol=0.0,
      jac=self.jac
    )
    return sol.y * const.UNA

  def solve_fom(
    self,
    t,
    n0,
    rtol=1e-6
  ):
    """Solve FOM."""
    n = self.solve(
      t=t,
      n0=n0,
      ops=self.fom_ops,
      rtol=rtol
    )
    return n[:1], n[1:]

  def solve_rom(
    self,
    t,
    n0,
    rtol=1e-6
  ):
    """Solve ROM."""
    self.check_rom_ops()
    self.is_einsum_used("solve_rom")
    # Encode initial condition
    z_m = self.encode(n0[1:])
    # Solve
    z = self.solve(
      t=t,
      n0=np.concatenate([n0[:1], z_m]),
      ops=self.rom_ops,
      rtol=rtol
    )
    # Decode solution
    n_m = self.decode(z[1:].T).T
    return z[:1], n_m

  def encode(self, x):
    return x @ self.psi

  def decode(self, z):
    return z @ self.phif.T

  # Data generation
  # ===================================
  def construct_design_mat(
    self,
    T_lim,
    p_lim,
    X_a_lim,
    nb_samples
  ):
    design_space = [np.sort(T_lim), np.sort(p_lim), np.sort(X_a_lim)]
    design_space = np.array(design_space).T
    # Construct
    ddim = design_space.shape[1]
    dmat = lhs(ddim, int(nb_samples))
    # Rescale
    amin, amax = design_space
    return dmat * (amax - amin) + amin

  def compute_fom_sol(
    self,
    t,
    mu,
    path=None,
    index=None,
    filename=None
  ):
    mui = mu[index] if (index is not None) else mu
    try:
      n0 = self.get_init_sol(*mui)
      n = self.solve_fom(t, n0)
      data = {"index": index, "mu": mui, "t": t, "n0": n0, "n": n}
      utils.save_case(path=path, index=index, data=data, filename=filename)
      converged = 1
    except:
      converged = 0
    return converged

  # Testing
  # ===================================
  def compute_rom_sol(
    self,
    path=None,
    index=None,
    filename=None,
    eval_err_on=None
  ):
    # Load test case
    icase = utils.load_case(path=path, index=index, filename=filename)
    n_fom, t, n0 = [icase[k] for k in ("n", "t", "n0")]
    # Solve ROM
    n_rom = self.solve_rom(t, n0)
    # Evaluate error
    if (eval_err_on == "mom"):
      # > Moments
      error = []
      for m in range(2):
        mom_rom = self.species["molecule"].compute_mom(n=n_rom[1], m=m)
        mom_fom = self.species["molecule"].compute_mom(n=n_fom[1], m=m)
        if (m == 0):
          mom0_fom = mom_fom
          mom0_rom = mom_rom
        else:
          mom_fom /= mom0_fom
          mom_rom /= mom0_rom
        error.append(utils.absolute_percentage_error(mom_rom, mom_fom))
      return np.vstack(error)
    elif (eval_err_on == "dist"):
      # > Distribution
      y_pred = n_rom[1] / const.UNA
      y_true = n_fom[1] / const.UNA
      return utils.absolute_percentage_error(y_pred, y_true, eps=1e-8)
    else:
      # > None: return the solution
      return t, n_fom, n_rom

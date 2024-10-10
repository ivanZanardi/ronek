import abc
import numpy as np
import scipy as sp

from pyDOE import lhs

from .. import const
from .. import utils
from .species import Species
from .kinetics import Kinetics
from typing import Dict, List, Optional, Tuple, Union


class BasicSystem(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    T: float,
    rates: str,
    species: Dict[str, str],
    use_einsum: bool = False,
    use_factorial: bool = False
  ) -> None:
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
    self._set_eq_ratio()
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
    self.rom_ops = None
    # Bases
    self.phi = None
    self.psi = None

  def _set_eq_ratio(self) -> None:
    q_a, q_m = [self.species[k].q_tot(self.T) for k in ("atom", "molecule")]
    self.gamma = q_m / q_a**2

  def _check_rom_ops(self) -> None:
    if (self.rom_ops is None):
      raise ValueError("Update ROM operators.")

  def _is_einsum_used(self, identifier: str) -> None:
    if self.use_einsum:
      raise NotImplementedError(
        "This functionality is not supported " \
        f"when using 'einsum': '{identifier}'."
      )

  # Operators
  # ===================================
  # FOM
  # -----------------------------------
  def update_fom_ops(self) -> None:
    # Update species
    for sp in self.species.values():
      sp.update(self.T)
    # Update kinetics
    self.kinetics.update(self.T)
    # Compose operators
    self._set_mass_matrix()
    self._set_mass_ratio()
    if self.use_einsum:
      self.fom_ops = self.kinetics.rates
    else:
      self.fom_ops = self._update_fom_ops(self.kinetics.rates)
    self.fom_ops["m_ratio"] = self.m_ratio

  def _set_mass_matrix(self) -> None:
    # Compose mass matrix
    m = np.full(self.nb_eqs, self.species["molecule"].m)
    m[0] = self.species["atom"].m
    self.M = np.diag(m)
    self.Minv = np.diag(1.0/m)

  def _set_mass_ratio(self) -> None:
    self.m_ratio = self.species["molecule"].m / self.species["atom"].m
    self.m_ratio = np.full(self.nb_eqs-1, self.m_ratio).reshape(1,-1)

  @abc.abstractmethod
  def _update_fom_ops(self, rates: dict) -> dict:
    pass

  # Linearized FOM
  # -----------------------------------
  def compute_lin_fom_ops(
    self,
    mu: np.ndarray,
    rho: float,
    max_mom: int = 10
  ) -> Dict[str, np.ndarray]:
    self._is_einsum_used("compute_lin_fom_ops")
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
  def update_rom_ops(
    self,
    phi: np.ndarray,
    psi: np.ndarray
  ) -> None:
    self._is_einsum_used("update_rom_ops")
    # Set basis
    self._set_basis(phi, psi)
    # Compose operators
    self.rom_ops = self._update_rom_ops()
    self.rom_ops["m_ratio"] = self.m_ratio @ self.phi

  def _set_basis(
    self,
    phi: np.ndarray,
    psi: np.ndarray
  ) -> None:
    self.phi, self.psi = phi, psi
    # Biorthogonalize
    self.phi = self.phi @ sp.linalg.inv(self.psi.T @ self.phi)
    # Check if complex
    if np.iscomplexobj(self.phi):
      self.phi = self.phi.real
    if np.iscomplexobj(self.psi):
      self.psi = self.psi.real

  @abc.abstractmethod
  def _update_rom_ops(self) -> None:
    pass

  # Physics
  # ===================================
  def compute_eq_comp(
    self,
    rho: float
  ) -> Tuple[np.ndarray]:
    # Solve this system of equations:
    # 1) rho_a + sum(rho_m) = rho
    # 2) n_m = gamma * n_a^2
    a = np.sum(self.gamma) * self.species["molecule"].m
    b = self.species["atom"].m
    c = -rho
    n_a = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    n_m = self.gamma*n_a**2
    return n_a.reshape(-1), n_m.reshape(-1)

  # Solving
  # ===================================
  # def get_init_sol(self, T=None, p=None, x_a=None, rho=None):
  #   if (p is None):
  #     return self._get_init_sol_from_rho(T, x_a, rho)
  #   else:
  #     return self._get_init_sol_from_p(T, x_a, p)

  # def _get_init_sol_from_rho(self, T, x_a, rho, noise=False, eps=1e-2):
  #   # Compute mass fractions
  #   # > Atom
  #   if noise:
  #     x_a = np.clip(x_a + eps * np.random.rand(1), 0, 1)
  #   norm = x_a * self.species["atom"].m \
  #        + (1-x_a) * self.species["molecule"].m
  #   w_a = x_a * self.species["atom"].m / norm
  #   # > Molecule
  #   q_m = self.species["molecule"].q_int(T)
  #   if noise:
  #     q_m *= (1 + eps * np.random.rand(*q_m.shape))
  #   w_m = (1-w_a) * (q_m / np.sum(q_m))
  #   # Compute number densities
  #   # > Atom
  #   n_a = rho * w_a / self.species["atom"].m
  #   n_a = np.array(n_a).reshape(-1)
  #   # > Molecule
  #   n_m = rho * w_m / self.species["molecule"].m
  #   return np.concatenate([n_a, n_m])

  def get_init_sol(
    self,
    T: float,
    w_a: float,
    rho: Optional[float] = None,
    noise: bool = False,
    sigma: float = 1e-2
  ) -> np.ndarray:
    # > Atom
    w_a = np.array(w_a).reshape(1)
    if noise:
      w_a = np.clip(w_a + sigma*np.random.rand(1), 0, 1)
    # > Molecule
    q_m = self.species["molecule"].q_int(T)
    if noise:
      q_m *= (1 + sigma*np.random.rand(*q_m.shape))
    w_m = (1-w_a)*(q_m / np.sum(q_m))
    w = np.concatenate([w_a, w_m])
    # Return mass fractions / number densities
    return w if (rho is None) else self._get_n(w, rho)

  def _get_w(
    self,
    n: np.ndarray,
    rho: float
  ) -> np.ndarray:
    w = (1/rho) * self.M @ n
    assert np.isclose(np.sum(w), 1.0, rtol=1e-5, atol=1e-8)
    return w

  def _get_n(
    self,
    w: np.ndarray,
    rho: float
  ) -> np.ndarray:
    return rho * self.Minv @ w

  def _get_rho(
    self,
    n: np.ndarray
  ) -> float:
    return np.diag(self.M) @ n

  # def _get_init_sol_from_p(self, T, x_a, p):
  #   n = p / (const.UKB * T)
  #   n_a = np.array([n * x_a]).reshape(-1)
  #   q_m = self.species["molecule"].q_int(T)
  #   n_m = n * (1-x_a) * q_m / np.sum(q_m)
  #   return np.concatenate([n_a, n_m])

  # Solving
  # ===================================
  def _solve(
    self,
    t: np.ndarray,
    n0: np.ndarray,
    ops: dict
  ) -> np.ndarray:
    sol = sp.integrate.solve_ivp(
      fun=self.fun,
      t_span=[0.0,t[-1]],
      y0=n0/const.UNA,
      method="LSODA",
      t_eval=t,
      args=(ops,),
      first_step=1e-14,
      rtol=1e-6,
      atol=0.0,
      jac=self.jac
    )
    return sol.y * const.UNA

  def solve_fom(
    self,
    t: np.ndarray,
    n0: np.ndarray
  ) -> np.ndarray:
    """Solve FOM."""
    n = self._solve(t=t, n0=n0, ops=self.fom_ops)
    return n[:1], n[1:]

  def solve_rom(
    self,
    t: np.ndarray,
    n0: np.ndarray
  ) -> np.ndarray:
    """Solve ROM."""
    self._check_rom_ops()
    self._is_einsum_used("solve_rom")
    # Encode initial condition
    z_m = self._encode(n0[1:])
    # Solve
    z = self._solve(
      t=t,
      n0=np.concatenate([n0[:1], z_m]),
      ops=self.rom_ops
    )
    # Decode solution
    n_m = self._decode(z[1:].T).T
    return z[:1], n_m

  def _encode(self, x: np.ndarray) -> np.ndarray:
    return x @ self.psi

  def _decode(self, z: np.ndarray) -> np.ndarray:
    return z @ self.phi.T

  def get_tgrid(
    self,
    start: float,
    stop: float,
    num: int
  ) -> np.ndarray:
    t = np.geomspace(start, stop, num=num-1)
    t = np.insert(t, 0, 0.0)
    return t

  # Data generation
  # ===================================
  def construct_design_mat(
    self,
    T_lim: List[float],
    w_a_lim: List[float],
    rho_lim: List[float],
    nb_samples: int
  ) -> np.ndarray:
    design_space = [np.sort(T_lim), np.sort(w_a_lim), np.sort(rho_lim)]
    design_space = np.array(design_space).T
    # Construct
    ddim = design_space.shape[1]
    dmat = lhs(ddim, int(nb_samples))
    # Rescale
    amin, amax = design_space
    return dmat * (amax - amin) + amin

  def compute_fom_sol(
    self,
    t: np.ndarray,
    mu: np.ndarray,
    path: Optional[str] = None,
    index: Optional[int] = None,
    filename: Optional[str] = None
  ) -> int:
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
    path: Optional[str] = None,
    index: Optional[int] = None,
    filename: Optional[str] = None,
    eval_err: Optional[str] = None,
    eps: float = 1e-8
  ) -> Union[np.ndarray, Tuple[np.ndarray]]:
    # Load test case
    icase = utils.load_case(path=path, index=index, filename=filename)
    n_fom, t, n0 = [icase[k] for k in ("n", "t", "n0")]
    # Solve ROM
    n_rom = self.solve_rom(t, n0)
    # Evaluate error
    if (eval_err == "mom"):
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
        error.append(utils.absolute_percentage_error(mom_rom, mom_fom, eps))
      return np.vstack(error)
    elif (eval_err == "dist"):
      # > Distribution
      rho = self._get_rho(n0)
      y_pred = n_rom[1] * self.species["molecule"].m / rho
      y_true = n_fom[1] * self.species["molecule"].m / rho
      return utils.absolute_percentage_error(y_pred, y_true, eps)
    else:
      # > None: return the solution
      return t, n_fom, n_rom

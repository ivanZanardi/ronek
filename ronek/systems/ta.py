import numpy as np

from .. import const
from .basic import BasicSystem
from typing import Dict


class TASystem(BasicSystem):
  """3-atomic system"""

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
    super(TASystem, self).__init__(
      T, rates, species, use_einsum, use_factorial
    )
    # Solving
    # -------------
    if self.use_einsum:
      self.fun = self._fun_einsum
    else:
      self.fun = self._fun_matmul
      self.jac = self._jac_matmul

  # Operators
  # ===================================
  # FOM
  # -----------------------------------
  def _update_fom_ops(self, rates):
    k = rates["m-a"]["e"]["fwd"] + rates["m-a"]["e"]["bwd"]
    k -= np.diag(np.sum(k, axis=-1) + rates["m-a"]["d"]["fwd"])
    return {
      "ed": np.transpose(k),
      "r": rates["m-a"]["d"]["bwd"]
    }

  # Linearized FOM
  # -----------------------------------
  def _compute_lin_fom_ops(
    self,
    mu: np.ndarray,
    rho: float,
    max_mom: int = 10
  ) -> Dict[str, np.ndarray]:
    # Equilibrium
    n_a_eq, n_m_eq = self.mix.compute_eq_comp(rho)
    n_eq = np.concatenate([n_a_eq, n_m_eq])
    w_eq = self.mix.get_w(n_eq, rho)
    # A operator
    A = self._compute_lin_fom_ops_a(n_a_eq)
    b = self._compute_lin_fom_ops_b(n_a_eq)
    A = np.hstack([b.reshape(-1,1), A])
    a = - self.mix.m_ratio @ A
    A = np.vstack([a.reshape(1,-1), A])
    A = self.mix.M @ A @ self.mix.Minv
    # C operator
    C = self._compute_lin_fom_ops_c(max_mom)
    # Initial solutions
    M = self._compute_lin_init_sols(mu, w_eq)
    # Return data
    return {"A": A, "C": C, "M": M, "x_eq": w_eq}

  def _compute_lin_fom_ops_a(
    self,
    n_a_eq: np.ndarray
  ) -> np.ndarray:
    return self.fom_ops["ed"] * n_a_eq

  def _compute_lin_fom_ops_b(
    self,
    n_a_eq: np.ndarray
  ) -> np.ndarray:
    return (
      self.fom_ops["ed"] @ self.mix.gamma \
      + 3 * self.fom_ops["r"]
    ) * n_a_eq**2

  def _compute_lin_fom_ops_c(self, max_mom):
    if (max_mom > 0):
      C = np.zeros((max_mom,self.nb_eqs))
      C[:,1:] = self.mix.species["molecule"].compute_mom_basis(max_mom)
    else:
      C = np.eye(self.nb_eqs)
      C[0,0] = 0.0
    return C

  def _compute_lin_init_sols(
    self,
    mu: np.ndarray,
    w_eq: np.ndarray,
    noise: bool = True,
    sigma: float = 1e-2
  ) -> np.ndarray:
    M = []
    for mui in mu:
      w0 = self.mix.get_init_sol(*mui, noise=noise, sigma=sigma)
      M.append(w0 - w_eq)
    M = np.vstack(M).T
    return M

  # ROM
  # -----------------------------------
  def _update_rom_ops(self):
    return {
      "ed": self.psi.T @ self.fom_ops["ed"] @ self.phi,
      "r": self.psi.T @ self.fom_ops["r"]
    }

  # Solving
  # ===================================
  def _fun_matmul(self, t, c, ops):
    # Extract number densities
    n = c * const.UNA
    n_a, n_m = n[:1], n[1:]
    # Compose source terms
    # > Molecule
    w_m = ops["ed"] @ n_m * n_a \
        + ops["r"] * n_a**3
    # > Atom
    w_a = - ops["m_ratio"] @ w_m
    # Compose full source term
    f = np.concatenate([w_a, w_m], axis=0)
    return f / const.UNA

  def _jac_matmul(self, t, c, ops):
    # Extract number densities
    n = c * const.UNA
    n_a, n_m = n[:1], n[1:]
    # Compose Jacobians
    # > Molecule
    j_ma = ops["ed"] @ n_m \
         + ops["r"] * 3*n_a**2
    j_mm = ops["ed"] * n_a
    j_m = np.concatenate([j_ma.reshape(-1,1), j_mm], axis=1)
    # > Atom
    j_a = - ops["m_ratio"] @ j_m
    # Compose full Jacobian
    j = np.concatenate([j_a, j_m], axis=0)
    return j

  def _fun_einsum(self, t, c, ops):
    # Extract variables
    n = c * const.UNA
    n_a, n_m = n[:1], n[1:]
    n_am = n_a * n_m
    # Compose source terms
    # > Molecule
    k_e = ops["m-a"]["e"]["fwd"] + ops["m-a"]["e"]["bwd"]
    w_m = - np.einsum("ij,i->i", k_e, n_am) \
        + np.einsum("ji,j->i", k_e, n_am) \
        - ops["m-a"]["d"]["fwd"] * n_am \
        + ops["m-a"]["d"]["bwd"] * (n_a**3)
    # > Atom
    w_a = - ops["m_ratio"] @ w_m
    # Compose full source term
    f = np.concatenate([w_a, w_m], axis=0)
    return f / const.UNA

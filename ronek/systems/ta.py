import numpy as np

from .. import const
from .basic import Basic


class TASystem(Basic):
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
      self.fom_fun = self._fom_fun_einsum
    else:
      self.fom_fun = self._fom_fun_matmul
      self.fom_jac = self._fom_jac_matmul

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
  def _compute_lin_fom_ops(self, Tint, max_mom=-1):
    return {
      "A": self.fom_ops["ed"] * const.UNA,
      "B": self._compute_lin_fom_ops_b(
        b=self._get_lin_fom_ops_b(),
        x0=self._get_lin_init_sols(Tint)
      ),
      "C": self._compute_lin_fom_ops_c(max_mom)
    }

  def _get_lin_fom_ops_b(self):
    b = self.fom_ops["ed"] @ self.gamma + 3*self.fom_ops["r"]
    return b * const.UNA**2

  def _get_lin_init_sols(self, Tint):
    return [self.species["molecule"].q_int(Ti) for Ti in Tint]

  def _compute_lin_fom_ops_b(self, b, x0):
    # Compose B
    B = np.vstack([b] + x0)
    # Normalize B
    B *= (np.linalg.norm(b) / np.linalg.norm(B, axis=-1, keepdims=True))
    return np.transpose(B)

  def _compute_lin_fom_ops_c(self, max_mom):
    if (max_mom > 0):
      return self.species["molecule"].compute_mom_basis(max_mom)
    else:
      return np.eye(self.species["molecule"].nb_comp)

  # ROM
  # -----------------------------------
  def _update_rom_ops(self):
    return {
      "ed": self.psi.T @ self.fom_ops["ed"] @ self.phif,
      "ed_eq": self.psi.T @ self.fom_ops["ed"] @ self.gamma,
      "r": self.psi.T @ self.fom_ops["r"]
    }

  # Solving
  # ===================================
  # FOM
  # -----------------------------------
  def _fom_fun_matmul(self, t, c, ops):
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

  def _fom_jac_matmul(self, t, c, ops):
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

  def _fom_fun_einsum(self, t, c, ops):
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

  # ROM
  # -----------------------------------
  def rom_fun(self, t, x, ops):
    # Extract variables
    x = x * const.UNA
    n_a, z = x[:1], x[1:]
    # Compose source terms
    # > Molecule
    w_m = ops["ed"] @ z * n_a \
        + ops["ed_eq"] * ops["n_a_eq"]**2 * n_a \
        + ops["r"] * n_a**3
    # > Atom
    w_a = - ops["m_ratio"] @ w_m
    # Compose full source term
    f = np.concatenate([w_a, w_m], axis=0)
    return f / const.UNA

  def rom_jac(self, t, x, ops):
    # Extract variables
    x = x * const.UNA
    n_a, z = x[:1], x[1:]
    # Compose Jacobians
    # > Molecule
    j_ma = ops["ed"] @ z \
         + ops["ed_eq"] * ops["n_a_eq"]**2 \
         + ops["r"] * 3*n_a**2
    j_mm = ops["ed"] * n_a
    j_m = np.concatenate([j_ma.reshape(-1,1), j_mm], axis=1)
    # > Atom
    j_a = - ops["m_ratio"] @ j_m
    # Compose full Jacobian
    j = np.concatenate([j_a, j_m], axis=0)
    return j

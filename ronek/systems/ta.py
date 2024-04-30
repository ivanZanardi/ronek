import numpy as np

from .. import const
from .basic import Basic


class TASystem(Basic):
  """3-atomic system"""

  # Initialization
  # ===================================
  def __init__(
    self,
    rates,
    species,
    use_einsum=False
  ):
    super(TASystem, self).__init__(rates, species, use_einsum)
    # Solving
    # -------------
    if self.use_einsum:
      self.fun = self._fun_einsum
    else:
      self.fun = self._fun_matmul
      self.jac = self._jac_matmul

  # Operators
  # ===================================
  # ROM
  # -----------------------------------
  def _update_rom_ops(self):
    return {
      "ed": self.psi.T @ self.fom_ops["ed"] @ self.phif,
      "r": self.psi.T @ self.fom_ops["r"]
    }

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
  def _compute_lin_fom_ops(self, T, Tint, max_mom=-1):
    alpha = self._compute_eq_ratio(T)
    return {
      "A": self.fom_ops["ed"] * const.UNA,
      "B": self._compute_lin_fom_ops_b(
        b=self._get_lin_fom_ops_b(alpha),
        Tint=Tint
      ),
      "C": self._compute_lin_fom_ops_c(max_mom)
    }

  def _get_lin_fom_ops_b(self, alpha):
    return (self.fom_ops["ed"] @ alpha + 3*self.fom_ops["r"]) * const.UNA**3

  def _compute_lin_fom_ops_b(self, b, Tint):
    B = np.vstack([b] + self._compute_boltz(Tint))
    # Normalize
    B *= (np.linalg.norm(b) / np.linalg.norm(B, axis=-1, keepdims=True))
    return np.transpose(B)

  def _compute_lin_fom_ops_c(self, max_mom):
    if (max_mom > 0):
      return self.species["molecule"].compute_mom_basis(max_mom)
    else:
      return np.eye(self.nb_eqs-1)

  # Solving
  # ===================================
  def _fun_matmul(self, t, c, ops):
    # Extract number densities
    n = c * const.UNA
    n_a, n_m = n[:1], n[1:]
    # Compose source terms
    # > Molecule
    w_m = (ops["ed"] @ n_m) * n_a + ops["r"] * (n_a**3)
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
    j_ma = ops["ed"] @ n_m + ops["r"] * (3*n_a**2)
    j_mm = ops["ed"] * n_a
    j_m = np.concatenate([j_ma.reshape(-1,1), j_mm], axis=1)
    # > Atom
    j_a = - ops["m_ratio"] @ j_m
    # Compose full Jacobian
    j = np.concatenate([j_a, j_m], axis=0)
    return j

  def _fun_einsum(self, t, c, ops):
    # Extract number densities
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

import torch
import numpy as np

from .. import const
from .ta import TASystem


class TAFASystem(TASystem):
  """3 and 4-atomic system"""

  # Initialization
  # ===================================
  def __init__(
    self,
    rates,
    species,
    use_einsum=False,
    use_factorial=False
  ):
    super(TAFASystem, self).__init__(rates, species, use_einsum, use_factorial)

  # Operators
  # ===================================
  # ROM
  # -----------------------------------
  def _update_rom_ops(self):
    ops_ma = self.fom_ops["m-a"]
    ops_mm = self.fom_ops["m-m"]
    return {
      # > Molecule-Atom collisions
      "m-a": {
        "ed_eq": self.psi.T @ ops_ma["ed"] @ self.gamma,
        "ed": self.psi.T @ ops_ma["ed"] @ self.phif,
        "r": self.psi.T @ ops_ma["r"]
      },
      # > Molecule-Molecule collisions
      "m-m": {
        "ed": np.einsum(
          "ip,ijk,jq,kr->pqr", self.psi, ops_mm["ed"], self.phif, self.phif
        ),
        "ed_eq_mat": self.psi.T @ self._compute_lin_fom_ops_a1() @ self.phif,
        "ed_eq_vec": self.psi.T @ (ops_mm["ed"] @ self.gamma @ self.gamma),
        "er_eq": self.psi.T @ ops_mm["er"] @ self.gamma,
        "er": self.psi.T @ ops_mm["er"] @ self.phif,
        "r": self.psi.T @ ops_mm["r"]
      }
    }

  # FOM
  # -----------------------------------
  def _update_fom_ops(self, rates):
    return {
      # > Molecule-Atom collisions
      "m-a": super(TAFASystem, self)._update_fom_ops(rates),
      # > Molecule-Molecule collisions
      "m-m": {
        "ed": self._update_fom_ops_e(rates["m-m"]["e"]) \
          + self._update_fom_ops_ed(rates["m-m"]["ed"]["fwd"]) \
          + self._update_fom_ops_d(rates["m-m"]["d"]["fwd"]),
        "er": self._update_fom_ops_er(rates["m-m"]["ed"]["bwd"]),
        "r": self._update_fom_ops_r(rates["m-m"]["d"]["bwd"])
      }
    }

  def _update_fom_ops_e(self, rates):
    rates = rates["fwd"] + rates["bwd"]
    # Diagonal
    k_diag = np.sum(rates, axis=(2,3))
    k_diag += np.transpose(k_diag)
    k_diag = torch.diag_embed(torch.from_numpy(k_diag))
    k_diag = np.transpose(k_diag.numpy(force=True), axes=(1,2,0))
    # Off-diagonal
    k = np.transpose(rates, axes=(2,3,0,1))
    k += np.transpose(k, axes=(1,0,2,3))
    k = np.sum(k, axis=0)
    return k - k_diag

  def _update_fom_ops_ed(self, rates):
    k = rates + np.transpose(rates, axes=(1,0,2))
    k = np.sum(k, axis=-1)
    k = torch.diag_embed(torch.from_numpy(k))
    k = np.transpose(k.numpy(force=True), axes=(1,2,0))
    return np.transpose(rates, axes=(2,0,1)) - k

  def _update_fom_ops_er(self, rates):
    k = rates + np.transpose(rates, axes=(0,2,1))
    k = np.transpose(np.sum(k, axis=-1))
    k -= np.diag(np.sum(rates, axis=(1,2)))
    return k

  def _update_fom_ops_d(self, rates):
    k = rates + np.transpose(rates)
    k = torch.diag_embed(torch.from_numpy(k))
    return - np.transpose(k.numpy(force=True), axes=(1,2,0))

  def _update_fom_ops_r(self, rates):
    k = rates + np.transpose(rates)
    return np.sum(k, axis=-1)

  # Linearized FOM
  # -----------------------------------
  def _compute_lin_fom_ops(self, rho, Tint, max_mom=10):
    n_a_eq, _ = self.compute_eq_comp(rho)
    return {
      "A": self._compute_lin_fom_ops_a(n_a_eq),
      "B": self._compute_lin_fom_ops_b(
        b=self._get_lin_fom_ops_b(n_a_eq),
        Tint=Tint
      ),
      "C": self._compute_lin_fom_ops_c(max_mom)
    }

  def _compute_lin_fom_ops_a(self, n_a_eq):
    A1 = self._compute_lin_fom_ops_a1()
    A2 = self.fom_ops["m-m"]["er"]
    A3 = self.fom_ops["m-a"]["ed"]
    if (n_a_eq == np.inf):
      A = A1 + A2
    elif (n_a_eq == 0.0):
      A = A3
    else:
      A = A1 + A2 + A3/n_a_eq
    return A * n_a_eq**2

  def _compute_lin_fom_ops_a1(self):
    A1 = self.fom_ops["m-m"]["ed"]
    return (A1 + np.transpose(A1, axes=(0,2,1))) @ self.gamma

  def _get_lin_fom_ops_b(self, n_a_eq):
    return (
      2 * (self.fom_ops["m-m"]["er"] @ self.gamma) * n_a_eq \
      + self.fom_ops["m-a"]["ed"] @ self.gamma \
      + 4 * self.fom_ops["m-m"]["r"] * n_a_eq \
      + 3 * self.fom_ops["m-a"]["r"]
    ) * n_a_eq**3 / const.UNA

  # Solving
  # ===================================
  def _fom_fun_matmul(self, t, c, ops):
    # Extract number densities
    n = c * const.UNA
    n_a, n_m = n[:1], n[1:]
    # Compose source terms
    # > Molecule
    w_m = ops["m-m"]["ed"] @ n_m @ n_m \
        + ops["m-m"]["er"] @ n_m * n_a**2 \
        + ops["m-a"]["ed"] @ n_m * n_a \
        + ops["m-m"]["r"] * n_a**4 \
        + ops["m-a"]["r"] * n_a**3
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
    j_ma = ops["m-m"]["er"] @ n_m * 2*n_a \
         + ops["m-a"]["ed"] @ n_m \
         + ops["m-m"]["r"] * 4*n_a**3 \
         + ops["m-a"]["r"] * 3*n_a**2
    j_mm = ops["m-m"]["ed"] @ n_m \
         + np.transpose(ops["m-m"]["ed"], axes=(0,2,1)) @ n_m \
         + ops["m-m"]["er"] * n_a**2 \
         + ops["m-a"]["ed"] * n_a
    j_m = np.concatenate([j_ma.reshape(-1,1), j_mm], axis=1)
    # > Atom
    j_a = - ops["m_ratio"] @ j_m
    # Compose full Jacobian
    j = np.concatenate([j_a, j_m], axis=0)
    return j

  def _fom_fun_einsum(self, t, c, ops):
    # Extract number densities
    n = c * const.UNA
    n_a, n_m = n[:1], n[1:]
    n_am = n_a * n_m
    # Compose source terms
    k_mm_e = ops["m-m"]["e"]["fwd"] + ops["m-m"]["e"]["bwd"]
    k_ma_e = ops["m-a"]["e"]["fwd"] + ops["m-a"]["e"]["bwd"]
    # > Molecule
    #   > Molecule-Molecule collisions
    w_m = - np.einsum("ijkl,i,j->i", k_mm_e, n_m, n_m) \
        - np.einsum("ijkl,i,j->j", k_mm_e, n_m, n_m) \
        + np.einsum("klij,k,l->i", k_mm_e, n_m, n_m) \
        + np.einsum("klij,k,l->j", k_mm_e, n_m, n_m) \
        - np.einsum("ijk,i,j->i", ops["m-m"]["ed"]["fwd"], n_m, n_m) \
        - np.einsum("ijk,i,j->j", ops["m-m"]["ed"]["fwd"], n_m, n_m) \
        + np.einsum("ijk,i,j->k", ops["m-m"]["ed"]["fwd"], n_m, n_m) \
        + np.einsum("kij,k->i", ops["m-m"]["ed"]["bwd"], n_m) * n_a**2 \
        + np.einsum("kij,k->j", ops["m-m"]["ed"]["bwd"], n_m) * n_a**2 \
        - np.einsum("kij,k->k", ops["m-m"]["ed"]["bwd"], n_m) * n_a**2 \
        - np.einsum("ij,i,j->i", ops["m-m"]["d"]["fwd"], n_m, n_m) \
        - np.einsum("ij,i,j->j", ops["m-m"]["d"]["fwd"], n_m, n_m) \
        + np.einsum("ij->i", ops["m-m"]["d"]["bwd"]) * n_a**4 \
        + np.einsum("ij->j", ops["m-m"]["d"]["bwd"]) * n_a**4
    #   > Molecule-Atom collisions
    w_m += - np.einsum("ij,i->i", k_ma_e, n_am) \
        + np.einsum("ji,j->i", k_ma_e, n_am) \
        - ops["m-a"]["d"]["fwd"] * n_am \
        + ops["m-a"]["d"]["bwd"] * n_a**3
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
    w_m = ops["m-m"]["ed"] @ z @ z \
        + ops["m-m"]["ed_eq_mat"] @ z * ops["n_a_eq"]**2 \
        + ops["m-m"]["ed_eq_vec"] * ops["n_a_eq"]**4 \
        + ops["m-m"]["er"] @ z * n_a**2 \
        + ops["m-m"]["er_eq"] * ops["n_a_eq"]**2 * n_a**2 \
        + ops["m-a"]["ed"] @ z * n_a \
        + ops["m-a"]["ed_eq"] * ops["n_a_eq"]**2 * n_a \
        + ops["m-m"]["r"] * n_a**4 \
        + ops["m-a"]["r"] * n_a**3
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
    j_ma = ops["m-m"]["er"] @ z * 2*n_a \
         + ops["m-m"]["er_eq"] * ops["n_a_eq"]**2 * 2*n_a \
         + ops["m-a"]["ed"] @ z \
         + ops["m-a"]["ed_eq"] * ops["n_a_eq"]**2 \
         + ops["m-m"]["r"] * 4*n_a**3 \
         + ops["m-a"]["r"] * 3*n_a**2
    j_mm = ops["m-m"]["ed"] @ z \
         + np.transpose(ops["m-m"]["ed"], axes=(0,2,1)) @ z \
         + ops["m-m"]["ed_eq_mat"] * ops["n_a_eq"]**2 \
         + ops["m-m"]["er"] * n_a**2 \
         + ops["m-a"]["ed"] * n_a
    j_m = np.concatenate([j_ma.reshape(-1,1), j_mm], axis=1)
    # > Atom
    j_a = - ops["m_ratio"] @ j_m
    # Compose full Jacobian
    j = np.concatenate([j_a, j_m], axis=0)
    return j

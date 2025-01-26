import os
import time
import torch
import numpy as np
import scipy as sp

from .. import utils
from .. import backend as bkd
from ronek.ops import svd_lowrank
from silx.io.dictdump import dicttoh5
from typing import Tuple, Optional


class CoBRAS(object):
  """
  Model Reduction for Nonlinear Systems by Balanced
  Truncation of State and Gradient Covariance (CoBRAS)

  See:
    https://doi.org/10.1137/22M1513228
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    system=None,
    quadrature=None,
    path_to_saving="./",
    saving=True,
    verbose=True
  ):
    self.verbose = verbose
    # System
    self.system = system
    self.quad = quadrature
    # Saving
    self.saving = saving
    self.path_to_saving = path_to_saving
    os.makedirs(self.path_to_saving, exist_ok=True)

  # Calling
  # ===================================
  def __call__(
    self,
    l: int,
    M: np.ndarray,
    xnot: Optional[list] = None,
    modes: bool = True,
    pod: bool = False
  ) -> None:
    X, Y = self.compute_cov_mats(l, M)
    if (xnot is not None):
      mask = self.make_mask(xnot)
      X, Y = X[mask], Y[mask]
    if modes:
      self.compute_modes(X, Y, pod)
    else:
      return X, Y

  def make_mask(self, xnot):
    xnot = np.array(xnot).astype(int).reshape(-1)
    mask = np.ones(self.system.nb_eqs)
    mask[xnot] = 0
    return mask.astype(bool)

  # Covariance matrices computation
  # -----------------------------------
  def compute_cov_mats(
    self,
    nb_meas: int = 10,     # number of output measuremtn (adjoint simulaiton)
    use_eig: bool = True,  # use eigedecomposiotn for deifinign max time for linear model validity
    err_max: float = 30.0  # percentrage error between linear and nonlinear model
  ) -> Tuple[np.ndarray, np.ndarray]:
    # Set quadrature points/weigths
    t, mu = [self.quad[k]["x"] for k in ("t", "mu")]
    w_t, w_mu = [self.quad[k]["w"] for k in ("t", "mu")]
    w_meas = 1.0 / np.sqrt(nb_meas)
    # Initialize dynamic arrays: state/gradient covariance matrices
    X, Y = [], []
    # Loop over 'forward problem' initial conditions (mu)
    # -------------
    for (i, mui) in enumerate(mu):
      # > Compute initial condition for 'forward problem'
      y0 = self.system.get_init_sol(mui, noise=True)
      # > Solve nonlinear 'forward problem'
      y = self.system.solve_fom(t, y0).T
      # > Compute time limits, valid for linear model
      tlim = self.system.compute_lin_tlim(t, y, use_eig, err_max)
      # > Get time grid for linear adjoint simulation
      tadj = self.system.get_tgrid(*tlim, nb_meas)
      # Loop over sampling initial times (t0)
      # -------------
      for j in range(len(t)):
        Yij = w_meas * self.solve_lin_adjoint(tadj, y[j]).T
        # > Quadrature weight
        wij = w_mu[i] * w_t[j]
        # > Store samples
        X.append(wij * y[j])
        Y.append(wij * Yij)
    # Return covariance matrices
    X = np.vstack(X).T
    Y = np.vstack(Y).T
    return X, Y

  # Linear adjoint model
  # -----------------------------------
  def solve_lin_adjoint(self, t, y0):
    # Compute linear operators
    self.system.compute_lin_fom_ops(0.0, y0)
    A, C = [getattr(self.system, k) for k in ("A", "C")]
    l, V = sp.linalg.eig(A)
    Vinv = sp.linalg.inv(V)
    # Allocate memory
    shape = [len(t)] + list(C.T.shape)
    g = np.zeros(shape)
    # Compute solution
    VC = V.T @ C.T
    for (i, ti) in enumerate(t):
      L = np.diag(np.exp(ti*l))
      g[i] = Vinv.T @ (L @ VC)
    # Manipulate tensor
    g = np.transpose(g, axes=(1,2,0))
    g = np.reshape(g, (shape[1],-1))
    return g

  # Balancing modes
  # -----------------------------------
  def compute_modes(
    self,
    X: np.ndarray,
    Y: np.ndarray,
    pod: bool = False,
    rank: int = 100,
    niter: int = 30
  ) -> None:
    if self.verbose:
      print("Computing CoBRAS modes ...")
    # Perform randomized SVD
    X, Y = [bkd.to_torch(z) for z in (X, Y)]
    U, s, V = svd_lowrank(
      X=X,
      Y=Y,
      q=min(rank, X.shape[0]),
      niter=niter
    )
    # Compute balancing transformation
    sqrt_s = torch.diag(torch.sqrt(1.0/s))
    phi = X @ V @ sqrt_s
    psi = Y @ U @ sqrt_s
    # Save balancing modes
    s, phi, psi = [bkd.to_numpy(z) for z in (s, phi, psi)]
    dicttoh5(
      treedict={"s": s, "phi": phi, "psi": psi},
      h5file=self.path_to_saving+"/cobras_bases.hdf5",
      overwrite_data=True
    )
    if pod:
      if self.verbose:
        print("Computing POD modes ...")
      U, s, _ = torch.svd_lowrank(
        A=X,
        q=min(rank, X.shape[0]),
        niter=niter
      )
      s, phi = [bkd.to_numpy(z) for z in (s, U)]
      dicttoh5(
        treedict={"s": s, "phi": phi, "psi": phi},
        h5file=self.path_to_saving+"/pod_bases.hdf5",
        overwrite_data=True
      )

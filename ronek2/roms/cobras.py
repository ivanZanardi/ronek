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
    l: int,
    M: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute covariance matrices for a given number of measurements and
    initial conditions.

    This function computes both the state and gradient covariance matrices by
    solving the forward and adjoint problems iteratively and returning the
    results based on the weighted samples.

    :param l: The number of measurements to consider.
    :type l: int
    :param M: Initial conditions matrix, where each row represents the initial
              state for the forward problem.
    :type M: np.ndarray

    :return: A tuple containing two covariance matrices:
             - X: State covariance matrix
             - Y: Gradient covariance matrix
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    tmax = 1e-7
    tmin = 1e-12
    nb_meas = 5
    fac = 1.0 / np.sqrt(nb_meas)
    # Set quadrature points
    t, mu = [self.quad[k]["x"] for k in ("t", "mu")]
    # Initialize dynamic arrays: state/gradient covariance matrices
    X, Y = [], []
    # Loop over 'forward problem' initial conditions (mu)
    # -------------
    for (i, mui) in enumerate(mu):
      # > Compute initial condition for 'forward problem'
      y0 = self.system.get_init_sol(mui)
      # > Solve 'forward problem'
      y = self.system.solve_fom(t, y0).T
      # Loop over sampling initial times (t0)
      # -------------
      for j in range(len(t)-1):
        t0 = max(t[j], tmin)
        tf = min(t[j+1], tmax)
        if t0 == 0.0:
          t = np.geomspace(t0, tf, num=nb_meas-1)
          t = np.insert(t, 0, 0.0)
        else:
        tj = np.geomspace(t0, tf, nb_meas)
        Yij = fac * self.solve_lin_adjoint(tj, y[j])
        # Store samples
        # -------------
        # Quadrature weight
        wij = self.quad["mu"]["w"][i] * self.quad["t"]["w"][j]
        X.append(wij * y[j])
        Y.append(wij * Yij)
    # Return covariance matrices
    X = np.vstack(X).T
    Y = np.vstack(Y).T
    return X, Y

  # Linear adjoint model
  # -----------------------------------
  def solve_lin_adjoint(self, t, y0):
    # Assembly linear operators
    C = self.system.C
    A, _ = self.system.get_lin_ops(y0)
    l, V = sp.linalg.eig(A)
    Vinv = sp.linalg.inv(V)
    # Allocate memory
    shape = [len(t)] + list(C.T.shape)
    g = np.zeros(shape)
    # Compute solution
    x = V.T @ C.T
    for (i, ti) in enumerate(t):
      Li = np.diag(np.exp(ti*l))
      g[i] = Vinv.T @ (Li @ x)
    # Manipulate tensor
    g = np.transpose(g, axes=(1,2,0))
    g = np.reshape(g, (shape[1],-1))
    return g

  def get_tgrid_adjoint(self, t0, tf, tmin, tmax, nb_pts):

    tf = min(tf, tmax)
    if t0 == 0.0:
      t = np.geomspace(tmin, tf, num=nb_pts-1)
      t = np.insert(t, 0, 0.0)
    else:
      t = np.geomspace(t0, tf, nb_pts)
    return t

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

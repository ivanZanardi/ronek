import os
import torch
import numpy as np
import scipy as sp

from . import backend as bkd
from silx.io.dictdump import dicttoh5, h5todict


class BPOD(object):
  """
  Model reduction using balanced proper orthogonal decomposition for
  a stable linear input-output system:
    dx = Ax + Bu
    y = Cx

  See:
    https://doi.org/10.1142/S0218127405012429
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    operators,
    lg_deg=5,
    path_to_saving="./",
    saving=True,
    verbose=True
  ):
    self.verbose = verbose
    # Initialize operators (A, B, and C)
    # -------------
    self.ops = operators
    # Nb. of equations
    self.nb_eqs = self.ops["A"].shape[0]
    # Gauss-Legendre quadrature points and weights
    # -------------
    self.lg_deg = lg_deg
    self.lg = np.polynomial.legendre.leggauss(self.lg_deg)
    self.lg = dict(zip(("x", "w"), self.lg))
    # Saving
    # -------------
    self.saving = saving
    self.path_to_saving = path_to_saving
    os.makedirs(self.path_to_saving, exist_ok=True)
    # Properties
    # -------------
    self.eiga = None

  # Properties
  # ===================================
  # Operators
  @property
  def ops(self):
    return self._ops

  @ops.setter
  def ops(self, value):
    self._ops = None
    if (value is not None):
      self._ops = {}
      for (k, op) in value.items():
        if (len(op.shape) == 1):
          op = op.reshape(-1,1)
        self._ops[k] = bkd.to_backend(op)

  # Eigendecomposition of operator A
  @property
  def eiga(self):
    return self._eiga

  @eiga.setter
  def eiga(self, value):
    self._eiga = None
    if (value is not None):
      self._eiga = {k: bkd.to_backend(x) for (k, x) in value.items()}

  # Calling
  # ===================================
  def __call__(
    self,
    t,
    X=None,
    Y=None,
    max_rank=0,
    real_only=True,
    compute_modes=True
  ):
    if ((X is None) or (Y is None)):
      self.set_quad(t)
      self.compute_eiga(real_only)
      if self.verbose:
        print("Computing Gramians ...")
      X, Y = self.compute_gramians(max_rank)
    if compute_modes:
      if self.verbose:
        print("Computing balancing modes ...")
      self.compute_balancing_modes(X, Y)
    else:
      return X, Y

  # Setting up
  # -----------------------------------
  # Gauss-Legendre quadrature
  def set_quad(self, t):
    _t, _w = [], []
    for i in range(len(t)-1):
      # Scaling and shifting
      a = (t[i+1] - t[i])/2
      b = (t[i+1] + t[i])/2
      wi = a * self.lg["w"]
      ti = a * self.lg["x"] + b
      _t.append(ti), _w.append(wi)
    # Set quadrature
    self.t, self.w = [
      bkd.to_backend(np.concatenate(z).squeeze()) for z in (_t, _w)
    ]
    self.time_dim = len(self.t)

  # Eigendecomposition
  def compute_eiga(self, real_only=True):
    if (self.eiga is None):
      filename = self.path_to_saving + "/eiga.hdf5"
      if os.path.exists(filename):
        eiga = h5todict(filename)
      else:
        if self.verbose:
          print("Performing eigendecomposition of A ...")
        a = bkd.to_numpy(self.ops["A"])
        l, v = sp.linalg.eig(a)
        vinv = sp.linalg.inv(v)
        eiga = {"l": l, "v": v, "vinv": vinv}
        # Save eigendecomposition
        if self.saving:
          dicttoh5(
            treedict=eiga,
            h5file=filename,
            overwrite_data=True
          )
      # Only real part
      if real_only:
        eiga = {k: x.real for (k, x) in eiga.items()}
      self.eiga = eiga

  # Gramians computation
  # -----------------------------------
  def compute_gramians(self, max_rank=0):
    # Compute the empirical controllability Gramian
    X = self.compute_gramian(op=self.ops["B"])
    # Compute the empirical observability Gramian
    if (max_rank > 0):
      # Perform output projection
      C = self.ops["C"].to("cpu")
      U = torch.linalg.svd(C @ X, full_matrices=False)[0]
      Ct = C.t() @ U[:,:max_rank]
      Ct = Ct.to(bkd.device())
    else:
      # Perform standard balanced truncation
      Ct = self.ops["C"].t()
    # Compute Gramian
    Y = self.compute_gramian(op=Ct, transpose=True)
    return [bkd.to_numpy(z) for z in (X, Y)]

  def compute_gramian(self, op, transpose=False):
    # Allocate memory
    shape = [self.time_dim] + list(op.shape)
    g = torch.zeros(shape, dtype=bkd.floatx(), device="cpu")
    # Compute tensor
    sqrt_w = torch.sqrt(self.w)
    y = self.eiga["v"].t() @ op if transpose else self.eiga["vinv"] @ op
    for i in range(self.time_dim):
      yi = y * torch.exp(self.t[i]*self.eiga["l"]).reshape(-1,1)
      gi = self.eiga["vinv"].t() @ yi if transpose else self.eiga["v"] @ yi
      g[i] = (sqrt_w[i] * gi).cpu()
    # Manipulate tensor
    g = torch.permute(g, dims=(1,2,0))
    g = torch.reshape(g, (self.nb_eqs,-1))
    return g

  # Balancing modes
  # -----------------------------------
  def compute_balancing_modes(self, X, Y):
    n, q, p = *X.shape, Y.shape[1]
    if (q*p > n**2):
      # Compute full Gramians
      WcWo = (X @ X.T) @ (Y @ Y.T)
      s, phi = sp.linalg.eig(WcWo)
      psi = sp.linalg.inv(phi).conj().T
      # Sorting
      indices = np.flip(np.argsort(s.real, axis=-1).reshape(-1))
      s, phi, psi = [np.take(x, indices, axis=-1) for x in (s, phi, psi)]
    else:
      # Perform SVD
      U, s, Vh = sp.linalg.svd(Y.T @ X, full_matrices=False)
      V = Vh.T
      # Compute balancing transformation
      sqrt_s = np.diag(np.sqrt(1/s))
      phi = X @ V @ sqrt_s
      psi = Y @ U @ sqrt_s
    # Save balancing modes
    dicttoh5(
      treedict={"s": s, "phi": phi, "psi": psi},
      h5file=self.path_to_saving+"/bases.hdf5",
      overwrite_data=True
    )

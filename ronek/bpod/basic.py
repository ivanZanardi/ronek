import os
import abc
import torch
import numpy as np
import scipy as sp
import tensorflow as tf

from .. import ops
from .. import backend as bkd
from silx.io.dictdump import dicttoh5, h5todict


class Basic(object):
  """
  Model reduction using balanced proper orthogonal decomposition for
  a stable input-output system.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    operators,
    lg_deg=5,
    path_to_save="./",
    saving=True
  ):
    # Initialize operators (A, B, and C)
    # -------------
    self.ops = operators
    for (k, op) in operators.items():
      if (len(op.shape) == 1):
        self.ops[k] = op.reshape(-1,1)
    # Nb. of equations
    self.fom_dim = self.ops["A"].shape[0]
    # Gauss-Legendre quadrature points and weights
    # -------------
    self.lg_deg = lg_deg
    self.lg = np.polynomial.legendre.leggauss(self.lg_deg)
    self.lg = dict(zip(("x", "w"), self.lg))
    # Saving
    # -------------
    self.path_to_save = path_to_save
    os.makedirs(self.path_to_save, exist_ok=True)
    self.saving = saving
    # Set properties
    # -------------
    self.eiga = None
    self.X = None
    self.Y = None

  # Properties
  # ===================================
  # Eigendecomposition of operator A
  @property
  def eiga(self):
    return self._eiga

  @eiga.setter
  def eiga(self, value):
    self._eiga = value

  # Empirical controllability Gramian
  @property
  def X(self):
    return self._X

  @X.setter
  def X(self, value):
    self._X = value

  # Empirical observability Gramian
  @property
  def Y(self):
    return self._Y

  @Y.setter
  def Y(self, value):
    self._Y = value

  # Calling
  # ===================================
  @abc.abstractmethod
  def __call__(self, *args, **kwargs):
    pass

  def prepare(self, t, real_only):
    # Get time grid and integration weights
    t, w = self.get_quad(t)
    self.time_dim = len(t)
    # Eigendecomposition of operator A
    self.compute_eiga(real_only)
    # Convert to backend
    self.eiga, self.ops, self.t, self.w = tf.nest.map_structure(
      bkd.to_backend, (self.eiga, self.ops, t, w)
    )

  # Gauss-Legendre quadrature
  # -----------------------------------
  def get_quad(self, t):
    _t, _w = [], []
    for i in range(len(t)-1):
      # Scaling and shifting
      a = (t[i+1] - t[i])/2
      b = (t[i+1] + t[i])/2
      wi = a * self.lg["w"]
      ti = a * self.lg["x"] + b
      _t.append(ti), _w.append(wi)
    # Concatenating
    t, w = [np.concatenate(z).squeeze() for z in (_t, _w)]
    return t, w

  # Eigendecomposition
  # -----------------------------------
  def compute_eiga(self, real_only=True):
    if (self.eiga is None):
      filename = self.path_to_save + "/eiga.hdf5"
      if os.path.exists(filename):
        # Read eigendecomposition
        eiga = h5todict(filename)
      else:
        # Perform eigendecomposition
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
  def compute_gc(self):
    if (self.X is None):
      self.X = ops.read(path=self.path_to_save, name="gc", to_cpu=True)
      if (self.X is None):
        self.X = self.compute_gramian(op=self.ops["B"])
        if self.saving:
          ops.save(self.X, filename=self.path_to_save+"/gc")

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
    g = torch.reshape(g, (self.fom_dim,-1))
    return g

  # Balancing modes
  # -----------------------------------
  def compute_balancing_modes(self, X, Y):
    # Compute full Gramians
    n, r = X.shape
    if (r > n):
      Wc, Wo = X @ X.t(), Y @ Y.t()
      s, T = torch.linalg.eig(Wc @ Wo)
      Tinv = torch.linalg.inv(V)
      self.s, indices = torch.sort(s, dim=-1, descending=True)
      self.phi = torch.take_along_dim(T, indices, dim=-1)
      self.psi = torch.take_along_dim(Tinv, indices, dim=-1)
    else:
      # Perform SVD
      U, self.s, Vh = torch.linalg.svd(Y.t() @ X, full_matrices=False)
      V = Vh.T
      # Compute balancing transformation
      sqrt_s = torch.diag(torch.sqrt(1/self.s))
      self.phi = X @ V @ sqrt_s
      self.psi = Y @ U @ sqrt_s
    # Save balancing modes
    if self.saving:
      for (x, name) in ((self.s, "s"), (self.phi, "phi"), (self.psi, "psi")):
        ops.save(x, filename=self.path_to_save+"/"+name)

import torch

from .. import ops
from .basic import Basic
from .. import backend as bkd


class Linear(Basic):
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
    path_to_save="./",
    saving=True
  ):
    super(Linear, self).__init__(
      operators=operators,
      lg_deg=lg_deg,
      path_to_save=path_to_save,
      saving=saving
    )

  # Calling
  # ===================================
  def __call__(
    self,
    t,
    real_only=True,
    maxrank=0,
    compute_modes=True
  ):
    print("Preparing ...")
    self.prepare(t, real_only)
    # Estimate Gramians (Eq. 20-21-22)
    print("Computing the empirical controllability Gramian ...")
    self.compute_gc()
    print("Computing the empirical observability Gramian ...")
    self.compute_go(maxrank)
    if compute_modes:
      # Compute balancing modes (Eq. 23-24)
      print("Computing the balancing modes ...")
      self.compute_balancing_modes(self.X, self.Y)

  # Gramians computation
  # -----------------------------------
  def compute_go(self, maxrank):
    if (self.Y is None):
      self.Y = ops.read(path=self.path_to_save, name="go", to_cpu=True)
      if (self.Y is None):
        if (maxrank > 0):
          # Perform output projection (Sec. 3.2)
          C = self.ops["C"].to("cpu")
          U = torch.linalg.svd(C @ self.X, full_matrices=False)[0]
          Ct = C.t() @ U[:,:maxrank]
        else:
          # Perform standard balanced truncation
          Ct = self.ops["C"].t()
        # Compute Gramian
        self.Y = self.compute_gramian(op=Ct.to(bkd.device()), transpose=True)
        if self.saving:
          ops.save(self.Y, filename=self.path_to_save+"/go")

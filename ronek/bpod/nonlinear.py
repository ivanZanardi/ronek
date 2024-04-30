import torch

from .. import ops
from tqdm import tqdm
from .basic import Basic
from .. import backend as bkd


class NonLinear(Basic):
  """
  Model reduction using balanced proper orthogonal decomposition for
  a stable linear system with nonlinear output:
    dx = Ax + Bu
    y = log(Cx)
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    operators,
    lg_deg=5,
    path_to_save="./"
  ):
    super(NonLinear, self).__init__(
      operators=operators,
      lg_deg=lg_deg,
      path_to_save=path_to_save
    )
    # Set properties
    self.scale = 1.0
    self.shift = 1.0

  # Properties
  # ===================================
  @property
  def scale(self):
    return self._scale

  @scale.setter
  def scale(self, value):
    self._scale = value

  @property
  def shift(self):
    return self._shift

  @shift.setter
  def shift(self, value):
    self._shift = value

  # Calling
  # ===================================
  def __call__(
    self,
    t,
    y0,
    trainer,
    scales,
    shift=1.0,
    real_only=True
  ):
    print("Preparing ...")
    self.prepare(t, real_only)
    print("Computing the empirical controllability Gramian ...")
    self.compute_gc()
    print("Computing the nonlinear empirical observability Gramian ...")
    self.compute_go(y0, trainer, scales, shift)
    print("Computing the balancing modes ...")
    self.compute_balancing_modes()

  # Gramians computation
  # -----------------------------------
  def compute_go(
    self,
    y0,
    trainer,
    scales,
    shift=1.0
  ):
    if (self.Y is None):
      self.Y = ops.read(path=self.path_to_save, name="go", to_cpu=True)
      if (self.Y is None):
        # Allocate memory
        shape = [scales.size, self.fom_dim]
        self.Y = torch.zeros(shape, dtype=bkd.floatx(), device="cpu")
        # Set shift and loop over scales
        self.shift = shift
        for (i, scale) in enumerate(tqdm(scales, ncols=80)):
          self.scale = scale
          self.Y[i] = torch.from_numpy(trainer(y0))
        self.Y = self.Y.t()
        ops.save(self.Y, filename=self.path_to_save+"/go")

  # Loss and gradient computation
  # -----------------------------------
  def cost(self, x):
    l = self.scale * self.compute_output(x) + self.shift
    l = 0.5*torch.log(l**2)
    l = torch.sum(torch.square(l), dim=-1)
    return - torch.sum(l * self.w)

  def euclidean_gradient(self, x):
    g = self.scale * self.compute_output(x) + self.shift
    g = torch.log(g) / g
    g = self.compute_output(g.t(), transpose=True)
    g *= (2 * self.scale * self.w.reshape(-1,1))
    return - torch.sum(g, dim=0)

  def compute_output(self, x, transpose=False):
    # Allocate memory
    shape = [self.time_dim, self.fom_dim]
    y = torch.zeros(shape, dtype=bkd.floatx(), device=bkd.device())
    # Compute projected vector
    # > Output matrix
    x = self.ops["C"].t() @ x if transpose else x
    # > Eigenvectors (Left/Right)
    x = self.eiga["v"].t() @ x if transpose else self.eiga["vinv"] @ x
    # > Loop over time instants
    is_1d = (len(x.shape) == 1)
    for i in range(self.time_dim):
      # > Eigenvalues
      si = torch.exp(self.t[i] * self.eiga["l"])
      yi = si*x if is_1d else si*x[:,i]
      # > Eigenvectors (Right/Left)
      yi = self.eiga["vinv"].t() @ yi if transpose else self.eiga["v"] @ yi
      # > Output matrix
      y[i] = yi if transpose else self.ops["C"] @ x
    # Return output
    return y

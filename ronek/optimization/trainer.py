import torch
import pymanopt

from .. import utils
from .. import backend as bkd
from .line_searcher import CustomAdaptiveLineSearcher


class Trainer(object):

  # Initialization
  # ===================================
  def __init__(self):
    # Control variables
    self.is_manifold_set = False
    self.is_problem_set = False
    self.is_optimizer_set = False

  # Setting
  # ===================================
  def set_problem(
    self,
    cost,
    euclidean_gradient,
    manifold={
      "type": "Sphere",
      "shape": []
    }
  ):
    self.manifold = utils.get_class(
      modules=[pymanopt.manifolds], name=manifold["type"]
    )(*manifold["shape"])
    self.is_manifold_set = True
    self.problem = pymanopt.Problem(
      manifold=self.manifold,
      cost=self.make_fun(cost),
      euclidean_gradient=self.make_fun(euclidean_gradient)
    )
    self.is_problem_set = True

  def set_optimizer(
    self,
    optimizer_kwargs={
      "max_iterations": 5,
      "min_step_size": 1e-20,
      "max_time": 1000,
      "log_verbosity": 1
    },
    line_searcher_kwargs={
      "contraction_factor": 0.5,
      "sufficient_decrease": 0.85,
      "max_iterations": 25,
      "initial_step_size": 1
    }
  ):
    # Set optimizer
    optimizer_kwargs["line_searcher"] = CustomAdaptiveLineSearcher(
      **line_searcher_kwargs
    )
    self.optimizer = pymanopt.optimizers.ConjugateGradient(**optimizer_kwargs)
    self.is_optimizer_set = True

  def make_fun(self, fun):
    if (not self.is_manifold_set):
      raise ValueError("Set the manifold using pymanopt.")
    fun_no_grad = torch.no_grad()(fun)
    fun_numpy = lambda x: bkd.to_numpy(fun_no_grad(bkd.to_backend(x)))
    return pymanopt.function.numpy(self.manifold)(fun_numpy)

  # Calling
  # ===================================
  def __call__(self, initial_point):
    if (not self.is_problem_set):
      raise ValueError("Set the problem using pymanopt.")
    if (not self.is_optimizer_set):
      raise ValueError("Set the optimizer using pymanopt.")
    return self.optimizer.run(
      problem=self.problem,
      initial_point=initial_point
    ).point

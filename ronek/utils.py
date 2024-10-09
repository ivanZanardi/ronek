import os
import sys
import types
import inspect
import collections
import numpy as np
import joblib as jl
import dill as pickle

from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union


# Classes
# =====================================
def get_class(
  modules: Union[types.ModuleType, List[types.ModuleType]],
  name: Union[str, None] = None,
  kwargs: Union[dict, None] = None
) -> callable:
  """
  Return a class object given its name and the module it belongs to.

  This function searches for a class with the specified name within the given
  module(s). If found, it can return an instance of the class with optional
  keyword arguments provided in `kwargs`. If no class is found, an error is
  raised.

  :param modules: A module or a list of modules to search for the class.
  :type modules: list or module
  :param name: The name of the class to retrieve.
  :type name: str, optional
  :param kwargs: Optional keyword arguments to pass when initializing the
                 class (it can contain the name of the class if 'name'
                 is not provided).
  :type kwargs: dict, optional

  :return: An instance of the class if found, or the class itself.
  :rtype: object or class
  """
  # Check class name
  if ((name is None) and (kwargs is not None)):
    if ("name" in kwargs.keys()):
      name = kwargs.pop("name")
    else:
      raise ValueError("Class name not provided.")
  # Loop over modules to find class
  if (not isinstance(modules, (list, tuple))):
    modules = [modules]
  for module in modules:
    members = inspect.getmembers(module, inspect.isclass)
    for (name_i, cls_i) in members:
      if (name_i == name):
        if (kwargs is not None):
          return cls_i(**kwargs)
        else:
          return cls_i
  # Raise error if class not found
  names = [module.__name__ for module in modules]
  raise ValueError(f"Class `{name}` not found in modules: {names}.")

def check_path(path: str) -> None:
  """
  Check if the specified path exists.

  :param path: The path to check.
  :type path: str

  :return: None
  :rtype: None

  :raises IOError: If the path does not exist.
  """
  if (not os.path.exists(path)):
    raise IOError(f"Path '{path}' does not exist.")

# Data
# =====================================
def save_case(
  path: str,
  index: int,
  data: Any,
  filename: Union[str, None] = None
) -> None:
  """
  Save simluated case to a file with specific format.

  The file is saved with a name formatted as `case_{index}.p`, where `index`
  is zero-padded to 5 digits.

  :param path: Directory path where the file will be saved.
  :type path: str
  :param index: Index used to generate the filename.
  :type index: int
  :param data: Data to be saved in the file.
  :type data: Any

  :return: None
  :rtype: None
  """
  if (filename is None):
    filename = path + f"/case_{str(index+1).zfill(4)}.p"
  pickle.dump(data, open(filename, "wb"))

def load_case(
  path: Union[str, None] = None,
  index: Union[int, None] = 0,
  key: Union[str, None] = None,
  filename: Union[str, None] = None
) -> Any:
  """
  Load simluated case from a file and optionally retrieve a specific item.

  Constructs the filename from the provided path and index, then loads
  the data from this file. If a key is specified, return the value associated
  with that key. Otherwise, return the entire data.

  :param path: Directory path where the case file is located.
  :type path: str
  :param index: Index used to generate the filename.
  :type index: int
  :param key: Optional key to retrieve a specific item from the data.
  :type key: Union[str, None]

  :return: The data from the file, or the specific item if a key is provided.
  :rtype: Any
  """
  if (filename is None):
    filename = path + f"/case_{str(index+1).zfill(4)}.p"
  if os.path.exists(filename):
    data = pickle.load(open(filename, "rb"))
    if (key is None):
      return data
    else:
      return data[key]

def load_case_parallel(
  path: str,
  ranges: List[int],
  key: Union[str, None] = None,
  nb_workers: int = 1
) -> List[Any]:
  """
  Load simluated cases in parallel or sequentially based on the number
  of workers.

  This function uses `joblib` to parallelize the loading of cases if
  `nb_workers` is greater than 1. Otherwise, it loads the cases sequentially.

  :param path: Path to the data source.
  :type path: str
  :param ranges: Range of indices for the cases to be loaded.
  :type ranges: List[int]
  :param key: Optional key to pass to the `load_case` function.
  :type key: Union[str, None]
  :param nb_workers: Number of parallel workers to use. Default is 1.
  :type nb_workers: int

  :return: A list of loaded cases.
  :rtype: List[Any]
  """
  iterable = tqdm(
    iterable=range(*ranges),
    ncols=80,
    desc="> Cases",
    file=sys.stdout
  )
  if (nb_workers > 1):
    return jl.Parallel(nb_workers)(
      jl.delayed(load_case)(path=path, index=i, key=key) for i in iterable
    )
  else:
    return [load_case(path=path, index=i, key=key) for i in iterable]

def generate_case_parallel(
  sol_fun: callable,
  nb_samples: int,
  sol_kwargs: Dict[str, Any] = {},
  nb_workers: int = 1,
  desc: str = "> Cases",
  verbose: bool = True
) -> None:
  """
  Generate cases in parallel and check solver convergence.

  The `sol_fun` callable function should return 0 or 1 to indicate whether
  the solver has converged or not.

  :param sol_fun: A callable that performs the solver operation and returns
                  convergence status as 0 or 1.
  :type sol_fun: callable
  :param nb_samples: Number of samples or cases to generate.
  :type nb_samples: int
  :param nb_workers: Number of parallel workers to use. Defaults to 1.
  :type nb_workers: int
  :param desc: Description to display in the progress bar. Defaults to
               "> Cases".
  :type desc: str
  :param verbose: If True, prints the total number of converged cases.
                  Defaults to True.
  :type verbose: bool

  :return: None
  :rtype: None

  This function uses `joblib` for parallel processing and `tqdm` for showing
  a progress bar. It applies the `sol_fun` function to a range of sample
  indices and collects convergence results. If `verbose` is True, it prints
  the total number of converged cases.
  """
  iterable = tqdm(
    iterable=range(nb_samples),
    ncols=80,
    desc=desc,
    file=sys.stdout
  )
  if (nb_workers > 1):
    converged = jl.Parallel(nb_workers)(
      jl.delayed(sol_fun)(index=i, **sol_kwargs) for i in iterable
    )
  else:
    converged = [sol_fun(index=i, **sol_kwargs) for i in iterable]
  if verbose:
    print(f"> Total converged cases: {sum(converged)}/{nb_samples}")

# Statistics
# =====================================
def absolute_percentage_error(y_true, y_pred, eps=1e-8):
  return 100*np.abs(y_true-y_pred)/(np.abs(y_true)+eps)

# Operations
# =====================================
def map_nested_dict(
  obj: Any,
  fun: callable
) -> Any:
  """
  Recursively apply a function to all values in a nested dictionary.

  This function traverses a nested dictionary and applies the given
  function to each value. It supports dictionaries, lists, and tuples.

  :param obj: The nested dictionary or other container to map.
  :type obj: dict or list or tuple or Any
  :param fun: The function to apply to each value.
  :type fun: Callable[[Any], Any]

  :return: A new nested structure with the function applied to all values.
  :rtype: Any
  """
  if isinstance(obj, collections.Mapping):
    return {k: map_nested_dict(v, fun) for (k, v) in obj.items()}
  else:
    if isinstance(obj, (list, tuple)):
      return [fun(x) for x in obj]
    else:
      return fun(obj)

# Integrals
# -------------------------------------
def get_gl_quad_1d(
  x: np.ndarray,
  deg: int = 3,
  dist: str = "uniform"
) -> Tuple[np.ndarray, np.ndarray]:
  """
  Compute 1D Gauss-Legendre quadrature points and weights over the domain
  defined by `x`.

  :param x: Array of points defining the intervals for quadrature.
  :type x: np.ndarray
  :param deg: Degree of the Gauss-Legendre quadrature (number of points
              within each subinterval). Default is 3.
  :type deg: int
  :param dist: The type of distribution to scale the quadrature weights.
               Default is "uniform".
  :type dist: str

  :raises ValueError: If the input `x` contains fewer than two points.

  :return: Tuple of arrays containing quadrature points and weights.
  :rtype: Tuple[np.ndarray, np.ndarray]
  """
  if (len(x) < 2):
    raise ValueError("The input must be at least of length 2.")
  # Limits
  a, b = np.amin(x), np.amax(x)
  # Compute Gauss-Legendre quadrature points
  # and weights for reference interval [-1, 1]
  xlg, wlg = np.polynomial.legendre.leggauss(deg)
  _x, _w = [], []
  # Loop over each interval in x
  for i in range(len(x) - 1):
    # Scaling and shifting from the reference
    # interval to the current interval
    a = 0.5 * (x[i+1] - x[i])
    b = 0.5 * (x[i+1] + x[i])
    _x.append(a * xlg + b)
    _w.append(a * wlg)
  # Concatenate all points and weights
  x = np.concatenate(_x).squeeze()
  w = np.concatenate(_w).squeeze()
  f = compute_dist(x, a, b, dist)
  return x, w*f

def get_gl_quad_2d(
  x: np.ndarray,
  y: np.ndarray,
  deg: int = 3,
  dist_x: str = "uniform",
  dist_y: str = "uniform"
) -> Tuple[np.ndarray, np.ndarray]:
  """
  Compute 2D Gauss-Legendre quadrature points and weights over the domain
  defined by `x` and `y`.

  :param x: Array of points defining the intervals along the x-axis.
  :type x: np.ndarray
  :param y: Array of points defining the intervals along the y-axis.
  :type y: np.ndarray
  :param deg: Degree of the Gauss-Legendre quadrature (number of points
              within each subinterval). Default is 3.
  :type deg: int
  :param dist_x: The type of distribution to scale the quadrature weights
                 along the x-axis. Default is "uniform".
  :type dist_x: str
  :param dist_y: The type of distribution to scale the quadrature weights
                 along the y-axis. Default is "uniform".
  :type dist_y: str

  :return: Tuple containing:
           - `xy`: (N, 2) array of quadrature points where N is
             the number of points.
           - `w`: Array of quadrature weights.
  :rtype: Tuple[np.ndarray, np.ndarray]
  """
  # Get 1D quadrature points and weights for x and y axes
  x, wx = get_gl_quad_1d(x, deg, dist_x)
  y, wy = get_gl_quad_1d(y, deg, dist_y)
  # Create 2D grid of points using meshgrid and reshape them into (N, 2)
  xy = [z.reshape(-1) for z in np.meshgrid(x, y)]
  xy = np.vstack(xy).T
  # Compute 2D quadrature weights by the product of the 1D weights
  w = [z.reshape(-1) for z in np.meshgrid(wx, wy)]
  w = np.prod(w, axis=0)
  return xy, w

def compute_dist(
  x: np.ndarray,
  a: float,
  b: float,
  model: str = "uniform"
) -> np.ndarray:
  """
  Compute the probability distribution over a set of points based on the
  specified distribution model.

  :param x: Array of points defining the intervals along the x-axis.
  :type x: np.ndarray
  :param model: The type of distribution to compute. Options are 'uniform' or
                'loguniform'. Default is 'uniform'.
  :type model: str

  :raises ValueError: If the specified model is not recognized.

  :return: Computed distribution values.
  :rtype: np.ndarray
  """
  if (model == "uniform"):
    return np.full(x.shape, 1/(b-a))
  elif (model == "loguniform"):
    dx = np.log(b) - np.log(a)
    return 1/(x*dx)
  else:
    raise ValueError(f"Distribution model not recognized: '{model}'")

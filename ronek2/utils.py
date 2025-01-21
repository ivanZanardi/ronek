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
    filename = path + f"/case_{str(index).zfill(4)}.p"
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
    filename = path + f"/case_{str(index).zfill(4)}.p"
  if os.path.exists(filename):
    data = pickle.load(open(filename, "rb"))
    if (key is None):
      return data
    else:
      return data[key]

def load_case_parallel(
  path: str,
  range: List[int],
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
  :param range: Range of indices for the cases to be loaded.
  :type range: List[int]
  :param key: Optional key to pass to the `load_case` function.
  :type key: Union[str, None]
  :param nb_workers: Number of parallel workers to use. Default is 1.
  :type nb_workers: int

  :return: A list of loaded cases.
  :rtype: List[Any]
  """
  range = np.sort(range)
  iterable = tqdm(
    iterable=range(*range),
    ncols=80,
    desc="Cases",
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
  irange: List[int],
  sol_kwargs: Dict[str, Any] = {},
  nb_workers: int = 1,
  desc: str = "Cases",
  verbose: bool = True,
  delimiter: str = "  "
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
  irange = np.sort(irange)
  iterable = tqdm(
    iterable=range(*irange),
    ncols=80,
    desc=delimiter+desc,
    file=sys.stdout
  )
  if (nb_workers > 1):
    runtime = jl.Parallel(nb_workers)(
      jl.delayed(sol_fun)(index=i, **sol_kwargs) for i in iterable
    )
  else:
    runtime = [sol_fun(index=i, **sol_kwargs) for i in iterable]
  runtime = [rt for rt in runtime if (rt is not None)]
  if verbose:
    nb_samples = irange[1]-irange[0]
    print(delimiter + f"Total converged cases: {len(runtime)}/{nb_samples}")
  return np.mean(runtime)

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

def is_nan_inf(x):
  return (np.isnan(x)+np.isinf(x)).astype(bool)

# Statistics
# =====================================
def absolute_percentage_error(y_true, y_pred, eps=1e-7):
  return 100*np.abs(y_true-y_pred)/(np.abs(y_true)+eps)

def l2_relative_error(y_true, y_pred, axis=-1, eps=1e-7):
  err = np.linalg.norm(y_true-y_pred, axis=axis)
  err /= (np.linalg.norm(y_true, axis=axis) + eps)
  return err

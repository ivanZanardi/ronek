import inspect


def get_class(
  modules=[],
  name=None,
  kwargs=None
):
  """
  Return a class object given its name and the module it belongs to.

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

  This function searches for a class with the specified name within the given
  module(s). If found, it can return an instance of the class with optional
  keyword arguments provided in `kwargs`. If no class is found, an error is
  raised.
  """
  # Check class name
  if (name is None) and (kwargs is not None):
    if ("name" in kwargs.keys()):
      name = kwargs.pop("name")
    else:
      raise ValueError("Class name not provided.")
  # Loop over modules to find class
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

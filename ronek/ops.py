import os
import torch
import pathlib
import numpy as np

from . import backend as bkd


def save(x, filename, to_numpy=False):
  # Convert to Numpy array
  if to_numpy:
    x = bkd.to_numpy(x)
  # Save tensor
  if torch.is_tensor(x):
    torch.save(obj=x, f=filename+".pt")
  else:
    with open(filename+".npy", "wb") as f:
      np.save(file=f, arr=x)

def read(path, name, to_cpu=False):
  file = [file for file in os.listdir(path) if file.startswith(name)]
  if (len(file) == 0):
    return None
  filename = path + "/" + file[0]
  if (pathlib.Path(filename).suffix == ".pt"):
    x = torch.load(filename)
    if to_cpu:
      x = x.cpu()
  else:
    x = np.load(filename)
  return x

def batched_matmul(a, b, nb_batches=1):
  ash, bsh = a.shape, b.shape
  if ((nb_batches > 1) and (len(ash) == 2) and (len(bsh) == 2)):
    # Set device
    refdev = a.device
    b = b.to(device=bkd.device())
    # Allocate memory
    shape = [int(ash[0]), int(bsh[-1])]
    y = torch.zeros(shape, dtype=bkd.floatx(), device=refdev)
    # Split indices
    indices = torch.arange(len(a)).to(device=refdev)
    indices = torch.tensor_split(indices, nb_batches)
    # Loop over batches
    for i in indices:
      ai = a[i].to(device=bkd.device())
      y[i] = torch.matmul(ai, b).to(device=refdev)
    return y
  else:
    return torch.matmul(a, b)

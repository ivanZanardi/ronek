# Environment
# =====================================
import sys
import importlib
if (importlib.util.find_spec("ronek") is None):
  sys.path.append("./../../../")

from ronek import env
env.set(
  device="cuda",
  device_idx=0,
  nb_threads=16,
  floatx="float64"
)

# Libraries
# =====================================
import numpy as np

from tqdm import tqdm
from ronek.systems import TAFASystem
from ronek.roms import BalancedTruncation


if (__name__ == '__main__'):

  # Inputs
  # ===================================
  # Time
  t_grid = {
    "lim": [1e-12, 1e-2],
    "pts": 50
  }
  # System
  # > Translational temperature
  T = 1e4
  # > Initial internal temperature (molecule)
  Tint_grid = {
    "lim": [3e2, 1e4],
    "pts": 50
  }
  # > Density
  rho_grid = {
    "lim": [1e-5, 1e0],
    "pts": 50
  }
  # > Moments of the distribution (molecule)
  max_mom = 10
  # Paths
  paths = {
    "dtb": "./../database/",
    "data": f"./data/max_mom_{max_mom}/"
  }

  # Isothermal master equation model
  # ===================================
  model = TAFASystem(
    rates=paths["dtb"] + "/kinetics.hdf5",
    species={
      k: paths["dtb"] + f"/species/{k}.json" for k in ("atom", "molecule")
    },
    use_einsum=False
  )
  model.update_fom_ops(T)
  model.set_eq_ratio(T)

  # Balanced truncation
  # ===================================
  # Time, density, and internal temperature grids
  t = model.get_tgrid(t_grid["lim"], num=t_grid["pts"])
  rho = np.geomspace(*rho_grid["lim"], num=rho_grid["pts"])
  Tint = np.geomspace(*Tint_grid["lim"], num=Tint_grid["pts"])
  # Model reduction
  X, Y = [], []
  for rhoi in tqdm(rho, ncols=80, desc="Densities"):
    # > Linear operators
    lin_ops = model.compute_lin_fom_ops(rho=rhoi, Tint=Tint, max_mom=max_mom)
    btrunc = BalancedTruncation(
      operators=lin_ops,
      lg_deg=3,
      path_to_saving=paths["data"],
      saving=False,
      verbose=False
    )
    # > Gramians
    Xi, Yi = btrunc(t=t, compute_modes=False)
    X.append(Xi), Y.append(Yi)
  # > Compute balancing modes
  X, Y = np.hstack(X), np.hstack(Y)
  btrunc.compute_balancing_modes(X, Y)

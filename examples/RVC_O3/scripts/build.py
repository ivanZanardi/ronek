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

from ronek.systems import TASystem
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
    "pts": 20
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
  model = TASystem(
    rates=paths["dtb"] + "/kinetics.hdf5",
    species={
      k: paths["dtb"] + f"/species/{k}.json" for k in ("atom", "molecule")
    },
    use_einsum=False,
    use_factorial=False
  )
  model.update_fom_ops(T)
  model.set_eq_ratio(T)

  # Balanced truncation
  # ===================================
  # Time and internal temperature grids
  t = model.get_tgrid(t_grid["lim"], num=t_grid["pts"])
  Tint = np.geomspace(*Tint_grid["lim"], num=Tint_grid["pts"])
  # Model reduction
  lin_ops = model.compute_lin_fom_ops(Tint=Tint, max_mom=max_mom)
  btrunc = BalancedTruncation(
    operators=lin_ops,
    lg_deg=3,
    path_to_saving=paths["data"]
  )
  btrunc(t)

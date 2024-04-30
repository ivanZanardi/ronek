import sys
sys.path.append("./../")

# Environment
# =====================================
from src import env
env.set(
  backend="torch",
  device="cuda",
  device_idx=1,
  nb_threads=16,
  epsilon=1e-10,
  floatx="float64"
)

# Libraries
# =====================================
import numpy as np

from src import bpod
from src.systems import TASystem

# Inputs
# =====================================
# System
T = 1e4
Tint_lim = [3e2, 1e4]
# Sizes of B and C
max_Tint = 10
max_mom = 10
# Paths
paths = {
  "dtb": "./../database/RVS_O3/",
  "data": "./../data/"
}

# Isothermal master equation model
# =====================================
model = TASystem(
  rates=paths["dtb"]+"kinetics.hdf5",
  species={k: paths["dtb"]+f"species/{k}.json" for k in ("atom", "molecule")},
  use_einsum=False
)
model.update_fom_ops(T)

# Balanced POD
# =====================================
# Time grid
t = np.geomspace(1e-12, 1e-2, num=50)
t = np.insert(t, 0, 0.0)
# Linear operators
Tint = np.linspace(*Tint_lim, num=max_Tint)
lin_ops = model.compose_lin_fom_ops(T=T, Tint=Tint, max_mom=max_mom)
# Model reduction
bpod_lin = bpod.Linear(
  operators=lin_ops,
  lg_deg=3,
  path_to_save=paths["data"]+f"/bpod_lin_ta.T{int(T)}K.m{max_mom}"
)
bpod_lin(t=t, real_only=True, maxrank=0)

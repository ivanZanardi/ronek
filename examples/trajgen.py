import os
import json
import torch
import numpy as np
import pandas as pd
import joblib as jl

from tqdm import tqdm
from pyDOE import lhs
from nitrom import backend as bkd
from silx.io.dictdump import dicttoh5
from hypernet.apps import utils as hyputils


class TrajGenerator(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    nb_traj,
    time_lim,
    time_pts,
    inp_space,
    paths={
      "physics": "./physics",
      "trajs": "./trajs"
    },
    nb_workers=4
  ):
    # Paths
    # ---------------------------------
    self.paths = {k: os.path.abspath(path) for (k, path) in paths.items()}
    os.makedirs(self.paths["trajs"], exist_ok=True)
    # Trajectories quantities
    # ---------------------------------
    self.nb_traj = nb_traj
    # Time grid
    self.time_vec = np.geomspace(*time_lim, num=time_pts-1)
    self.time_vec = np.array([0.0] + self.time_vec.tolist())
    # Set input space to be sampled
    self.inp_space = np.vstack(list(inp_space.values())).T
    self.inp_space_lbl = list(inp_space.keys())
    # Initializing thermophysical models
    # -----------------------------------
    with open(self.paths["physics"] + "/inputs.json") as file:
      phyinp = json.load(file)
    # Species thermo
    self.sp_th = hyputils._get_specie_thermos(phyinp)
    # Mixture
    self.mix = hyputils._get_mixture(phyinp, self.sp_th)
    # Kinetics
    self.kin = hyputils._get_kinetics(phyinp, self.sp_th, self.mix)
    # Species
    for st in self.sp_th.values():
      if (st.specie.nb_comp > 1):
        self.molecule = st.specie.name
      else:
        self.atom = st.specie.name
    # Multi-threading
    # ---------------------------------
    self.nb_workers = nb_workers

  # Sample initial conditions
  # ===================================
  def sample_inp_space(
    self,
    read_samples=False
  ):
    filename = self.paths["trajs"] + "/cases.csv"
    if read_samples:
      x = pd.read_csv(filename).to_dict(orient="list")
      x = {k: np.array(v) for (k, v) in x.items()}
    else:
      # Contruct design matrix (DM)
      d = self.inp_space.shape[1]
      x = lhs(d, self.nb_traj)
      # Rescale DM
      x = x * (self.inp_space[1] - self.inp_space[0]) + self.inp_space[0]
      # Restructure DM
      x = [xi.squeeze() for xi in np.split(x, x.shape[-1], axis=-1)]
      x = {k: x[i] for (i, k) in enumerate(self.inp_space_lbl)}
      # Save DM
      xdf = pd.DataFrame.from_dict(x)
      xdf.to_csv(path_or_buf=filename, index=False, float_format="%.8e")
    # Manipulate DM
    return self._manipulate_inp_space(x)

  def _manipulate_inp_space(
    self,
    samples
  ):
    samples = {
      k: bkd.to_backend(x.reshape(-1,1)) for (k, x) in samples.items()
    }
    # Set initial composition
    samples[f"X_{self.molecule}"] = 1 - samples[f"X_{self.atom}"]
    self.mix.init_mix(
      x={k: samples[f"X_{k}"] for k in self.sp_th.keys()},
      var="X",
      p_mix=samples["p"],
      T=samples["Ti"],
      flatten=False
    )
    # Collect initial composition
    init_comp = torch.hstack([st.specie.n for st in self.sp_th.values()])
    return bkd.to_numpy(init_comp)

  # Generate trajectories
  # ===================================
  def generate_trajs(
    self,
    model,
    samples,
    start=0
  ):
    iterable = tqdm(
      iterable=range(start, len(samples)),
      ncols=80,
      desc="Box cases"
    )
    sols = jl.Parallel(self.nb_workers)(
      jl.delayed(model.solve)(t=self.time_vec, y0=samples[i]) \
        for i in iterable
    )
    sols = {
      str(i+1).zfill(5): {"t": self.time_vec, "y": sols[i]} \
        for i in range(start, len(samples))
    }
    # Save solutions
    dicttoh5(
      treedict=sols,
      h5file=self.paths["trajs"] + "/sols.hdf5",
      overwrite_data=True,
      create_dataset_args={"compression": "gzip"}
    )

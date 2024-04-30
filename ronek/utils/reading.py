import numpy as np
import pandas as pd

from scipy.constants import physical_constants as pc
UNA = pc["Avogadro constant"][0]
UKB = pc["Boltzmann constant"][0]


# Reading
# =====================================
def read_lev_sol(lev_file, path_to_sol, molecule):
  lev = read_lev(filename=lev_file)
  sol = read_box(filename=path_to_sol+"/box.dat")
  # Read population
  sol[f"n_{molecule}"] = read_pop(
    filename=path_to_sol + f"/pop_{molecule}.dat",
    nb_instants=len(sol["t"]),
    nb_lev=lev["nb"]
  ) * lev["g"].reshape(1,-1)
  return lev, sol

def read_box(filename):
  data = pd.read_csv(
    filename, delimiter="  ", skiprows=1, engine="python"
  )
  data.columns = ["t", "X_O", "X_O2", "T", "rho", "p", "n", "E"]
  data = data.to_dict("list")
  data = {k: np.array(v) for k, v in data.items()}
  return data

def read_pop(filename, nb_instants, nb_lev):
  ni_ov_g = np.loadtxt(filename, comments="&", skiprows=2, usecols=(1))
  ni_ov_g = np.abs(ni_ov_g.reshape((nb_instants, nb_lev)))
  return ni_ov_g

def read_lev(filename):
  dtb = pd.read_csv(filename)
  dtb["EVib"] = dtb["EVib"] - np.amin(dtb["EVib"])
  dtb["ERot"] = dtb["ERot"] - np.amin(dtb["ERot"])
  dtb["E"] = dtb["E"] - np.amin(dtb["E"])
  dtb = dtb.sort_values(by="E")
  dtb = dtb.to_dict("list")
  dtb = {k: np.array(v) for k, v in dtb.items()}
  dtb["nb"] = len(dtb["g"])
  return dtb

import os
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt

from .. import const
from .. import utils

COLORS = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


# Plotting
# =====================================
# Cumulative energy
def plot_cum_energy(
  y,
  figname=None,
  save=False,
  show=True
):
  # Initialize figure
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  x = np.arange(1,len(y)+1)
  ax.set_xlabel("$i$-th basis")
  # y axis
  ax.set_ylabel("Cumulative energy")
  # Plotting
  ax.plot(x, y, marker="o")
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()

# Time evolution
def plot_evolution(
  x,
  y,
  ls=None,
  ylim=None,
  labels=[r"$t$ [s]", r"$n$ [m$^{-3}$]"],
  scales=["log", "linear"],
  legend_loc="best",
  figname=None,
  save=False,
  show=False
):
  # Initialize figures
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  ax.set_xlabel(labels[0])
  ax.set_xscale(scales[0])
  ax.set_xlim([np.amin(x), np.amax(x)])
  # y axis
  ax.set_ylabel(labels[1])
  ax.set_yscale(scales[1])
  if (ylim is not None):
    ax.set_ylim(ylim)
  # Plotting
  if isinstance(y, dict):
    i = 0
    for (k, yk) in y.items():
      if ("FOM" in k.upper()):
        _c = "k"
        _ls = "-" if (ls is None) else ls
      else:
        _c = COLORS[i]
        _ls = "--" if (ls is None) else ls
        i += 1
      ax.plot(x, yk, ls=_ls, c=_c, label=k)
    ax.legend(loc=legend_loc)
  else:
    ax.plot(x, y, "-", c="k")
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()

# Moments
def plot_mom_evolution(
  path,
  t,
  n_m,
  molecule,
  molecule_label,
  err_scale="linear",
  max_mom=2
):
  path = path + "/moments/"
  os.makedirs(path, exist_ok=True)
  # Compute moments
  moms = [{k: molecule.compute_mom(nk, m=0) for (k, nk) in n_m.items()}]
  for m in range(1,max_mom):
    moms.append(
      {k: molecule.compute_mom(nk, m=1)/moms[0][k] for (k, nk) in n_m.items()}
    )
  # Plot moments
  for m in range(max_mom):
    if (m == 0):
      yscale = "log"
      label_sol = fr"$n_{molecule_label}$ [m$^{{-3}}$]"
      label_err = fr"$n_{molecule_label}$ error [\%]"
    else:
      yscale = "linear"
      if (m == 1):
        label_sol = fr"$e_{molecule_label}$ [eV]"
        label_err = fr"$e_{molecule_label}$ error [\%]"
      else:
        label_sol = fr"$\gamma_{m}$ [eV$^{m}$]"
        label_err = fr"$\gamma_{m}$ error [\%]"
    # > Moment
    plot_evolution(
      x=t,
      y=moms[m],
      labels=[r"$t$ [s]", label_sol],
      legend_loc="lower left" if (m == 0) else "upper left",
      scales=["log", yscale],
      figname=path + f"/m{m}",
      save=True,
      show=False
    )
    # > Moment error
    for (k, momk) in moms[m].items():
      if ("FOM" in k.upper()):
        mom_fom = momk
        break
    moms_err = {}
    for (k, momk) in moms[m].items():
      if ("FOM" not in k.upper()):
        moms_err[k] = utils.absolute_percentage_error(momk, mom_fom)
    plot_evolution(
      x=t,
      y=moms_err,
      labels=[r"$t$ [s]", label_err],
      scales=["log", err_scale],
      figname=path + f"/m{m}_err",
      save=True,
      show=False
    )

def plot_err_evolution(
  path,
  err,
  eval_err_on,
  molecule_label,
  err_scale="linear",
  max_mom=2
):
  os.makedirs(path, exist_ok=True)
  for r in err.keys():
    t = err[r]["t"]
    break
  if (eval_err_on == "mom"):
    for m in range(max_mom):
      if (m == 0):
        label = fr"$n_{molecule_label}$ error [\%]"
      else:
        if (m == 1):
          label = fr"$e_{molecule_label}$ error [\%]"
        else:
          label = fr"$\gamma_{m}$ error [\%]"
      plot_evolution(
        x=t,
        y={f"$r={r}$": err[r]["mean"][m] for r in err.keys()},
        ls="-",
        labels=[r"$t$ [s]", label],
        scales=["log", err_scale],
        figname=path + f"/mean_m{m}_err",
        save=True,
        show=False
      )
  else:
    plot_evolution(
      x=t,
      y={f"$r={r}$": err[r]["mean"] for r in err.keys()},
      ls="-",
      labels=[r"$t$ [s]", r"$n_i/g_i$ error [\%]"],
      scales=["log", err_scale],
      figname=path + "/mean_dist_err",
      save=True,
      show=False
    )

# 2D distribution
def as_si(x, ndp=0):
  s = "{x:0.{ndp:d}e}".format(x=x, ndp=ndp)
  m, e = s.split("e")
  return r"{m:s}\times 10^{{{e:d}}}".format(m=m, e=int(e))

def plot_dist_2d(
  x,
  y,
  t=None,
  labels=[r"$\epsilon_i$ [eV]", r"$n_i/g_i$ [m$^{-3}$]"],
  scales=["linear", "log"],
  markersize=6,
  figname=None,
  save=False,
  show=False
):
  # Initialize figures
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  ax.set_xlabel(labels[0])
  ax.set_xscale(scales[0])
  # y axis
  ax.set_ylabel(labels[1])
  ax.set_yscale(scales[1])
  # Plotting
  style = dict(
    linestyle="",
    marker="o"
  )
  if isinstance(y, dict):
    i = 0
    lines = []
    for (k, yk) in y.items():
      if ("FOM" in k.upper()):
        c = "k"
        ymin = yk.min()
      else:
        c = COLORS[i]
        i += 1
      yk[yk<ymin*0.1] = 0.0
      ax.plot(x, yk, c=c, markersize=markersize, **style)
      lines.append(plt.plot([], [], c=c, **style)[0])
    ax.legend(lines, list(y.keys()))
  else:
    ax.plot(x, y, c="k", markersize=markersize, **style)
  if (t is not None):
    ax.text(
      0.05, 0.05,
      r"$t = {0:s}$ s".format(as_si(t)),
      transform=ax.transAxes,
      fontsize=25
    )
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()

def plot_multi_dist_2d(
  path,
  teval,
  t,
  n_m,
  molecule,
  markersize=6,
  shift_diss_en=False
):
  path = path + "/dist/"
  os.makedirs(path, exist_ok=True)
  # Interpolate at "teval" points
  n_m_eval = {}
  for (k, nk) in n_m.items():
    if (nk.shape[-1] != molecule.nb_comp):
      nk = nk.T
    nk = sp.interpolate.interp1d(t, nk, kind="cubic", axis=0)(teval)
    n_m_eval[k] = nk / molecule.lev["g"]
  if shift_diss_en:
    x = (molecule.lev["e"] - molecule.e_d) / const.eV_to_J
  else:
    x = molecule.lev["e"] / const.eV_to_J
  for i in range(len(teval)):
    plot_dist_2d(
      x=x,
      y={k: nk[i] for (k, nk) in n_m_eval.items()},
      t=float(teval[i]),
      labels=[r"$\epsilon_i$ [eV]", r"$n_i/g_i$ [m$^{-3}$]"],
      scales=["linear", "log"],
      markersize=markersize,
      figname=path + f"/t{i+1}",
      save=True,
      show=False
    )

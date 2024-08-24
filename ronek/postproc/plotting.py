import numpy as np
import matplotlib
import matplotlib.pyplot as plt

COLORS = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]


# Plotting
# =====================================
# 2D distribution
def dist_2d(
  x,
  y,
  y_pred=None,
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
  # > FOM
  labels = [y[0]]
  lines = [plt.plot([], [], c="k", **style)[0]]
  ax.plot(x, y[1], c="k", markersize=markersize, **style)
  # > ROMs
  if (y_pred is not None):
    for (i, yi) in enumerate(y_pred):
      labels.append(yi[0])
      lines.append(plt.plot([], [], c=COLORS[i], **style)[0])
      ax.plot(x, yi[1], c=COLORS[i], markersize=markersize, **style)
    ax.legend(lines, labels)
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()

# Cumulative energy
def cum_energy(
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
def evolution(
  x,
  y,
  y_pred=None,
  labels=[r"$t$ [s]", r"$n$ [m$^{-3}$]"],
  scales=["log", "linear"],
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
  # Plotting
  # > FOM
  labels = [y[0]]
  lines = [plt.plot([], [], "-", c="k")[0]]
  ax.plot(x, y[1], "-", c="k")
  # > ROMs
  if (y_pred is not None):
    for (i, yi) in enumerate(y_pred):
      labels.append(yi[0])
      lines.append(plt.plot([], [], "--", c=COLORS[i])[0])
      ax.plot(x, y_pred, "--", c=COLORS[i])
    ax.legend(lines, labels)
  # Tight layout
  plt.tight_layout()
  if save:
    plt.savefig(figname)
  if show:
    plt.show()
  plt.close()

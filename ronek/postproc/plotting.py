import numpy as np

from matplotlib import pyplot as plt


# Plotting
# =====================================
style = dict(
  markeredgewidth=1,
  lw=1.0,
  linestyle="",
  markersize=2.0,
  marker="x"
)

# 2D distribution
def dist_2d(
  x,
  y,
  y_pred=None,
  labels=[r"$\epsilon_i$ [eV]", r"$n_i/g_i$ [m$^{-3}$]"],
  scales=["linear", "log"],
  filename=None,
  save=False,
  show=False
):
  # Initialize figures
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  ax.set_xlabel(labels[0])
  ax.set_yscale(scales[0])
  # y axis
  ax.set_ylabel(labels[1])
  ax.set_yscale(scales[1])
  # Plotting
  ax.plot(x, y, c="k", **style)
  if (y_pred is not None):
    ax.plot(x, y_pred, c="r", **style)
  if save:
    plt.savefig(filename)
  if show:
    plt.show()
  plt.close()

# Cumulative energy
def cum_energy(
  y,
  filename=None,
  save=False,
  show=True
):
  # Initialize figure
  fig = plt.figure()
  ax = fig.add_subplot()
  # x axis
  x = np.arange(1,len(y)+1)
  ax.set_xlabel("$k$-th basis")
  # y axis
  ax.set_ylabel("Cumulative energy")
  # Plotting
  ax.plot(
    x,
    y,
    markeredgewidth=1,
    lw=1.0,
    markersize=5.0,
    marker="o"
  )
  if save:
    plt.savefig(filename)
  if show:
    plt.show()
  plt.close()

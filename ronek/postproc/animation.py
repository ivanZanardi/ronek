from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


# Animation
# =====================================
# Initialize lines for levels distribution
def _init_lines(
  ax,
  markersize
):
  # Set up axes
  ax.set_xlabel(r'$\epsilon_i$ [eV]')
  ax.set_ylabel(r'$n_i\//\/g_i$ [m$^{-3}$]')
  # Initialize lines
  style = dict(
    linestyle="",
    marker="o",
    fillstyle='full'
  )
  lines = []
  for c in ("k", "r"):
    lines.append(ax.semilogy([], [], c=c, markersize=markersize, **style)[0])
  # Add legend
  ax.legend(
    [ax.semilogy([], [], c=c, **style)[0] for c in ("k", "r")],
    labels=["FOM", "ROM"],
    loc='lower left'
  )
  return ax, lines

# Create animation
def _create_animation(
  t,
  x,
  y,
  frames,
  markersize
):
  # Initialize a figure in which the graphs will be plotted
  fig, ax = plt.subplots()
  # Initialize levels distribution lines objects
  ax, lines = _init_lines(ax, markersize)
  # Initialize text in ax
  txt = ax.text(0.7, 0.92, '', transform=ax.transAxes, fontsize=25)
  # Tight layout
  plt.tight_layout()

  def _animate(frame):
    i = int(frame/frames*len(t))
    # Write time instant
    txt.set_text(r'$t$ = %.1e s' % t[i])
    # Loop over models
    for (j, k) in enumerate(("FOM", "ROM")):
      lines[j].set_data(x, y[k][i])
    # Rescale axis limits
    ax.relim()
    ax.autoscale_view(tight=True)
    return lines

  # Get animation
  return FuncAnimation(
    fig,
    _animate,
    frames=frames,
    blit=True
  )

def animate(
  t,
  x,
  y,
  markersize=6,
  frames=10,
  fps=10,
  filename="./lev_dist.gif",
  dpi=600,
  save=True,
  show=False
):
  # Create animation
  anim = _create_animation(t, x, y, frames, markersize)
  # Save animation
  if save:
    anim.save(filename, fps=fps, dpi=dpi)
  # Display animation
  if show:
    HTML(anim.to_jshtml())

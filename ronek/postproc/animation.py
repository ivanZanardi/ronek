from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


# Animation
# =====================================
# Initialize lines for levels distribution
def _init_lines(
  ax
):
  # Set up axes
  ax.set_xlabel(r'$\epsilon_i$ [eV]')
  ax.set_ylabel(r'$n_i\//\/g_i$ [m$^{-3}$]')
  # Initialize lines
  style = dict(
    lw=1,
    marker='x',
    fillstyle='full',
    markersize=0.3,
    linestyle=''
  )
  lines = []
  for c in ("k", "r"):
    lines.append(ax.semilogy([], [], c=c, **style)[0])
  # Add legend
  ax.legend(lines, labels=["FOM", "ROM"], loc='lower left')
  return ax, lines

# Create animation
def _create_animation(
  t,
  x,
  y,
  frames
):
  # Initialize a figure in which the graphs will be plotted
  fig, ax = plt.subplots()
  # Initialize levels distribution lines objects
  ax, lines = _init_lines(ax)
  # Initialize text in ax
  txt = ax.text(0.77, 0.92, '', transform=ax.transAxes, fontsize=10)

  def _animate(frame):
    idx = int(frame/frames*len(t))
    # Write time instant
    txt.set_text(r'$t$ = %.1e s' % t[idx])
    # Loop over models
    for (i, k) in enumerate(("FOM", "ROM")):
      lines[i].set_data(x, y[k])
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
  frames=10,
  fps=10,
  filename="./lev_dist.gif",
  dpi=600,
  save=True,
  show=False
):
  # Create animation
  anim = _create_animation(t, x, y, frames)
  # Save animation
  if save:
    anim.save(filename, fps=fps, dpi=dpi)
  # Display animation
  if show:
    HTML(anim.to_jshtml())

import numpy as np

from ronek import const
from ronek.postproc import plotting as pltt
from ronek.postproc import animation as anim


# Solving
# =====================================
def get_tgrid(t_lim, num):
  t = np.geomspace(*t_lim, num=num-1)
  t = np.insert(t, 0, 0.0)
  return t

def get_y0(model, T, p, Xa):
  n = p / (const.UKB * T)
  na = np.array([n * Xa]).reshape(-1)
  qm = model.species["molecule"].q_int(T)
  nm = n * (1-Xa) * qm / np.sum(qm)
  return na, nm

def solve_fom(model, t, na_0, nm_0):
  y0 = np.concatenate([na_0, nm_0])
  y = model.solve(t, y0, ops=model.fom_ops, rtol=1e-7, atol=0.0)
  return y[:1], y[1:]

def solve_rom(model, t, na_0, nm_0, phi, psi, rom_dim):
  # Update operators
  model.set_basis(phi=phi[:,:rom_dim], psi=psi[:,:rom_dim])
  model.update_rom_ops()
  # Solve
  y0 = np.concatenate([na_0, model.psi.T @ nm_0])
  y = model.solve(t, y0, ops=model.rom_ops, rtol=1e-7, atol=0.0)
  return y[:1], model.phi @ y[1:]

# Plotting
# =====================================
def plot_moments(path, t, yfom, yrom, ei, max_mom=2):
  for m in range(max_mom):
    mom_fom = np.sum(yfom*ei**m, axis=0)
    mom_rom = np.sum(yrom*ei**m, axis=0)
    if (m == 0):
      label = r"$n$ [m$^{-3}$]"
      mom0_fom, mom0_rom = mom_fom, mom_rom
    else:
      label = r"$e$ [eV]" if (m == 1) else fr"$\gamma_{m}$ [eV$^{m}$]"
      mom_fom /= mom0_fom
      mom_rom /= mom0_rom
    # Plot moment
    pltt.evolution(
      x=t,
      y=mom_fom,
      y_pred=mom_rom,
      labels=[r"$t$ [s]", label],
      scales=["log", "log"],
      figname=path + f"/mom_{m}.png",
      save=True,
      show=False
    )
    # Plot moment error
    mom_err = 100 * np.abs(mom_rom - mom_fom) / np.abs(mom_fom)
    pltt.evolution(
      x=t,
      y=mom_err,
      labels=[r"$t$ [s]", fr"Error $\gamma_{m}$ [%]"],
      scales=['log', 'linear'],
      figname=path + f"/mom_{m}_err.png",
      save=True,
      show=False
    )

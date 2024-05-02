import numpy as np

from ronek import const


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

def solve_fom(model, t, y0):
  y0 = np.concatenate(y0)
  y = model.solve(t, y0, ops=model.fom_ops, rtol=1e-6, atol=0.0)
  return y[:1], y[1:]

def solve_rom(model, t, y0, phi, psi, rom_dim, abs=False):
  na_0, nm_0 = y0
  # Update operators
  model.set_basis(phi=phi[:,:rom_dim], psi=psi[:,:rom_dim])
  model.update_rom_ops()
  # Solve
  y0 = np.concatenate([na_0, model.psi.T @ nm_0])
  y = model.solve(t, y0, ops=model.rom_ops, rtol=1e-6, atol=0.0)
  if abs:
    return y[:1], np.abs(model.phi @ y[1:])
  else:
    return y[:1], model.phi @ y[1:]

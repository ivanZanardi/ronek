import numpy as np

from ronek import const


def get_tgrid(t_lim, num):
  t = np.geomspace(*t_lim, num=num-1)
  t = np.insert(t, 0, 0.0)
  return t

def get_n(model, T, p, Xa):
  n = p / (const.UKB * T)
  na = np.array([n * Xa]).reshape(-1)
  qm = model.species["molecule"].q_int(T)
  nm = n * (1-Xa) * qm / np.sum(qm)
  return np.concatenate([na, nm])

def solve_fom(model, t, y0):
  y = model.solve(t, y0, ops=model.fom_ops, rtol=1e-6, atol=0.0)
  return y[:1], y[1:]

def solve_rom(model, t, y0, phi, psi):
  # Update operators
  model.set_basis(phi, psi)
  model.update_rom_ops()
  # Solve
  y = model.solve_rom(t, y0, rtol=1e-6, atol=0.0)
  return y[:1], y[1:]

import abc
import numpy as np
import scipy as sp

from pyDOE import lhs

from .. import const
from .. import utils
from .species import Species
from .kinetics import Kinetics


class Basic(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    T,
    rates,
    species,
    use_einsum=False,
    use_factorial=False
  ):
    # Thermochemistry database
    # -------------
    self.T = float(T)
    # > Species
    self.nb_eqs = 0
    self.species = {}
    for k in ("atom", "molecule"):
      if (k not in species):
        raise ValueError(
          "The 'species' input parameter should be a " \
          "dictionary with 'atom' and 'molecule' as keys."
        )
      self.species[k] = Species(species[k], use_factorial)
      self.nb_eqs += self.species[k].nb_comp
    self.set_eq_ratio()
    # > Kinetics
    self.kinetics = Kinetics(rates, self.species)
    # FOM
    # -------------
    # Solving
    self.use_einsum = use_einsum
    self.fom_fun = None
    self.fom_jac = None
    # Operators
    ratio = self.species["molecule"].m / self.species["atom"].m
    self.mass_ratio = np.full((1,self.nb_eqs-1), ratio)
    self.update_fom_ops()
    # ROM
    # -------------
    self.rom_ops = None
    # Bases
    self.phi = None
    self.psi = None
    self.phif = None

  def set_eq_ratio(self):
    q_a, q_m = [self.species[k].q_tot(self.T) for k in ("atom", "molecule")]
    self.gamma = q_m / q_a**2

  def check_rom_ops(self):
    if (self.rom_ops is None):
      raise ValueError("Update ROM operators.")

  def is_einsum_used(self, identifier):
    if self.use_einsum:
      raise NotImplementedError(
        "This functionality is not supported " \
        f"when using 'einsum': '{identifier}'."
      )

  # Operators
  # ===================================
  # FOM
  # -----------------------------------
  def update_fom_ops(self):
    # Update species
    for sp in self.species.values():
      sp.update(self.T)
    # Update kinetics
    self.kinetics.update(self.T)
    # Compose operators
    if self.use_einsum:
      self.fom_ops = self.kinetics.rates
    else:
      self.fom_ops = self._update_fom_ops(self.kinetics.rates)
    self.fom_ops["m_ratio"] = self.mass_ratio

  @abc.abstractmethod
  def _update_fom_ops(self, rates):
    pass

  # Linearized FOM
  # -----------------------------------
  def compute_lin_fom_ops(self, *args, **kwargs):
    self.is_einsum_used("compute_lin_fom_ops")
    return self._compute_lin_fom_ops(*args, **kwargs)

  @abc.abstractmethod
  def _compute_lin_fom_ops(self, *args, **kwargs):
    pass

  # ROM
  # -----------------------------------
  def update_rom_ops(self, phi, psi, biortho=True):
    self.is_einsum_used("update_rom_ops")
    # Set basis
    self.set_basis(phi, psi, biortho)
    # Compose operators
    self.rom_ops = self._update_rom_ops()
    self.rom_ops["m_ratio"] = self.mass_ratio @ self.phif

  def set_basis(self, phi, psi, biortho=True):
    self.phi, self.phif, self.psi = phi, phi, psi
    # Biorthogonalize
    if biortho:
      self.phif = self.phi @ sp.linalg.inv(self.psi.T @ self.phi)
    # Check if complex
    for k in ("phi", "psi", "phif"):
      bases = getattr(self, k)
      if np.iscomplexobj(bases):
        setattr(self, k, bases.real)

  @abc.abstractmethod
  def _update_rom_ops(self):
    pass

  @abc.abstractmethod
  def rom_fun(self, t, x, ops):
    pass

  @abc.abstractmethod
  def rom_jac(self, t, x, ops):
    pass

  # Equilibrium composition
  # -----------------------------------
  def compute_rho(self, n):
    n_a, n_m = n[:1], n[1:]
    rho = n_a * self.species["atom"].m \
        + np.sum(n_m) * self.species["molecule"].m
    return rho

  def compute_eq_comp(self, rho):
    # Solve this system of equations:
    # 1) rho_a + sum(rho_m) = rho
    # 2) n_m = gamma * n_a^2
    a = np.sum(self.gamma) * self.species["molecule"].m
    b = self.species["atom"].m
    c = -rho
    n_a = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    n_m = self.gamma*n_a**2
    return n_a, n_m

  def compute_eq_dist(self, Tint):
    q = [self.species["molecule"].q_int(Ti) for Ti in Tint]
    return [qi/np.sum(qi) for qi in q]

  # Solving
  # ===================================
  def get_init_sol(self, T, p, X_a):
    n = p / (const.UKB * T)
    n_a = np.array([n * X_a]).reshape(-1)
    q_m = self.species["molecule"].q_int(T)
    n_m = n * (1-X_a) * q_m / np.sum(q_m)
    return np.concatenate([n_a, n_m])

  def get_tgrid(self, start, stop, num):
    t = np.geomspace(start, stop, num=num-1)
    t = np.insert(t, 0, 0.0)
    return t

  def solve(
    self,
    t,
    y0,
    fun,
    jac=None,
    ops=None,
    rtol=1e-6
  ):
    if (ops is None):
      raise ValueError("Provide set of operators as input.")
    sol = sp.integrate.solve_ivp(
      fun=fun,
      t_span=[0.0,t[-1]],
      y0=y0/const.UNA,
      method="LSODA",
      t_eval=t,
      args=(ops,),
      first_step=1e-14,
      rtol=rtol,
      atol=0.0,
      jac=jac
    )
    return sol.y * const.UNA

  def solve_fom(
    self,
    t,
    n0,
    rtol=1e-6,
    *args,
    **kwargs
  ):
    """Solve state-to-state FOM."""
    n = self.solve(
      t=t,
      y0=n0,
      fun=self.fom_fun,
      jac=self.fom_jac,
      ops=self.fom_ops,
      rtol=rtol
    )
    return n[:1], n[1:]

  def solve_rom_cg_m0(
    self,
    t,
    n0,
    rtol=1e-6,
    *args,
    **kwargs
  ):
    """Solve coarse-graining-based ROM."""
    self.check_rom_ops()
    self.is_einsum_used("solve_rom_cg_m0")
    # Encode initial condition
    z_m = self.encode(n0[1:])
    # Solve
    z = self.solve(
      t=t,
      y0=np.concatenate([n0[:1], z_m]),
      fun=self.fom_fun,
      jac=self.fom_jac,
      ops=self.rom_ops,
      rtol=rtol
    )
    # Decode solution
    n_m = self.decode(z[1:].T).T
    return z[:1], n_m

  def solve_rom_bt(
    self,
    t,
    n0,
    rtol=1e-6,
    use_abs=False,
    *args,
    **kwargs
  ):
    """Solve balanced truncation-based ROM."""
    self.check_rom_ops()
    self.is_einsum_used("solve_rom_bt")
    # Compute equilibrium value
    rho = self.compute_rho(n=n0)
    n_a_eq, n_m_eq = self.compute_eq_comp(rho)
    self.rom_ops["n_a_eq"] = n_a_eq
    # Encode initial condition
    z_m = self.encode(n0[1:], x_eq=n_m_eq)
    # Solve
    z = self.solve(
      t=t,
      y0=np.concatenate([n0[:1], z_m]),
      fun=self.rom_fun,
      jac=self.rom_jac,
      ops=self.rom_ops,
      rtol=rtol
    )
    # Decode solution
    n_m = self.decode(z[1:].T, x_eq=n_m_eq).T
    if use_abs:
      n_m = np.abs(n_m)
    return z[:1], n_m

  def encode(self, x, x_eq=None):
    if (x_eq is not None):
      x = x - x_eq
    return x @ self.psi

  def decode(self, z, x_eq=None):
    x = z @ self.phif.T
    if (x_eq is not None):
      x = x + x_eq
    return x

  # Data generation
  # ===================================
  def construct_design_mat(
    self,
    T_lim,
    p_lim,
    X_a_lim,
    nb_samples
  ):
    design_space = [np.sort(T_lim), np.sort(p_lim), np.sort(X_a_lim)]
    design_space = np.array(design_space).T
    # Construct
    ddim = design_space.shape[1]
    dmat = lhs(ddim, int(nb_samples))
    # Rescale
    amin, amax = design_space
    return dmat * (amax - amin) + amin

  def compute_fom_sol(
    self,
    t,
    mu,
    path=None,
    index=None,
    filename=None
  ):
    mui = mu[index] if (index is not None) else mu
    try:
      n0 = self.get_init_sol(*mui)
      n = self.solve_fom(t, n0)
      data = {"index": index, "mu": mui, "t": t, "n0": n0, "n": n}
      utils.save_case(path=path, index=index, data=data, filename=filename)
      converged = 1
    except:
      converged = 0
    return converged

  # Testing
  # ===================================
  def compute_rom_sol(
    self,
    model="bt",
    path=None,
    index=None,
    filename=None,
    eval_err_on=None
  ):
    # Load test case
    icase = utils.load_case(path=path, index=index, filename=filename)
    n_fom, t, n0 = [icase[k] for k in ("n", "t", "n0")]
    # Solve ROM
    solve = self.solve_rom_bt if (model == "bt") else self.solve_rom_cg_m0
    n_rom = solve(t, n0)
    # Evaluate error
    if (eval_err_on == "mom"):
      # > Moments
      error = []
      for m in range(2):
        mom_rom = self.species["molecule"].compute_mom(n=n_rom[1], m=m)
        mom_fom = self.species["molecule"].compute_mom(n=n_fom[1], m=m)
        if (m == 0):
          mom0_fom = mom_fom
          mom0_rom = mom_rom
        else:
          mom_fom /= mom0_fom
          mom_rom /= mom0_rom
        error.append(utils.absolute_percentage_error(mom_rom, mom_fom))
      return error
    elif (eval_err_on == "dist"):
      # > Distribution
      y_pred = n_rom[1] / const.UNA
      y_true = n_fom[1] / const.UNA
      return utils.absolute_percentage_error(y_pred, y_true, eps=1e-8)
    else:
      # > None: return the solution
      return t, n_fom, n_rom

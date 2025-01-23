import torch
import numpy as np
import scipy as sp

from .. import const
from .. import backend as bkd
from .argoncr import ArgonCR
from typing import Dict, Optional


class Equilibrium(object):

  # Compute equilibirum state (mass fractions and tempruatere)

  # Solve this system of equations:
  # -------------
  # 1) Charge neutrality:
  #    x_em = x_Arp
  # 2) Mole conservation:
  #    x_em + x_Arp + x_Ar = 1
  # 3) Detailed balance:
  #    (n_Arp*n_em)/n_Ar = (Q_Arp*Q_em)/Q_Ar = keq
  #    from reaction:
  #    Ar <-> Ar^+ + e^-
  # -------------

  # Initialization
  # ===================================
  def __init__(
    self,
    system: ArgonCR,
    clipping: bool = True
  ) -> None:
    self.system = system
    self.clipping = clipping
    self.lsq_opts = dict(
      method="trf",
      ftol=1e-08,
      xtol=1e-08,
      gtol=0.0,
      max_nfev=int(1e5)
    )
    self.set_fun_jac()

  def set_fun_jac(self):
    for name in ("from_prim", "from_cons", "full_system"):
      # Function
      fun = getattr(self, f"_{name}_fun")
      setattr(self, f"{name}_fun", bkd.make_fun_np(fun))
      # Jacobian
      jac = torch.func.jacrev(fun, argnums=0)
      setattr(self, f"{name}_jac", bkd.make_fun_np(jac))

    #   setattr(self, name[1:], bkd.make_fun_np(jac))
    # # Primitive variables
    # self.from_prim_fun = bkd.make_fun_np(self._from_prim_fun)
    # self.from_prim_jac = torch.func.jacrev(self._from_prim_fun)
    # self.from_prim_jac = bkd.make_fun_np(self.from_prim_jac)
    # # Primitive variables - Thermal nonequilibrium
    # self.from_prim_th_neq_fun = bkd.make_fun_np(self._from_prim_th_neq_fun)
    # self.from_prim_th_neq_jac = torch.func.jacrev(self._from_prim_th_neq_fun)
    # self.from_prim_th_neq_jac = bkd.make_fun_np(self.from_prim_th_neq_jac)
    # # Conservative variables
    # self.from_cons_fun = bkd.make_fun_np(self._from_cons_fun)
    # self.from_cons_jac = torch.func.jacrev(self._from_cons_fun, argnums=0)
    # self.from_cons_jac = bkd.make_fun_np(self.from_cons_jac)

  # Primitive variables
  # ===================================
  def from_prim(
    self,
    rho: float,
    T: float,
    Te: Optional[float] = None
  ) -> np.ndarray:
    """Compute equilibirum state from primritive macrosocpiv variables,
    such as density, temperarue and electron temperatue.
    """
    # Convert to 'torch.Tensor'
    rho, T, Te = [bkd.to_torch(z).reshape(1) for z in (rho, T, Te)]
    # Update mixture
    self.system.mix.set_rho(rho)
    self.system.mix.update_species_thermo(T, Te)
    # Compute electron molar fraction
    x = sp.optimize.least_squares(
      fun=self.from_prim_fun,
      x0=np.log([1e-4]),
      jac=self.from_prim_jac,
      bounds=(-np.inf, 0.0),
      **self.lsq_opts
    ).x
    x_em = bkd.to_torch(np.exp(x))
    x_em = self._clipping(x_em)
    # Update composition
    self._update_composition(x_em)
    # Compute number densities
    w = self.system.mix.get_qoi_vec("w")
    y = torch.cat([w, T, Te])
    return bkd.to_numpy(y)

  def _from_prim_fun(self, x: torch.Tensor) -> torch.Tensor:
    # Extract variables
    x_em = torch.exp(x)
    # Update composition
    self._update_composition(x_em)
    # Enforce detailed balance
    return self._detailed_balance()

  # Primitive variables - Thermal nonequilibrium
  # ===================================
  def from_prim_th_neq(
    self,
    rho: float,
    T: float,
    Te: float
  ) -> np.ndarray:
    """Compute equilibirum state from primritive macrosocpiv variables,
    such as density, temperarue and electron temperatue, assuming with thermal nonequlibiurm meaning T != Te
    """
    self.system.isothermal = True
    # Initial guess
    y0 = self.from_prim(rho, T, Te)
    # Set bounds
    bound_min = np.full(len(y0), -np.inf)
    bound_max = np.full(len(y0), 0.0)
    bound_max[-2:] = np.log(1e5)
    # Compute electron molar fraction
    x = sp.optimize.least_squares(
      fun=self.full_system_fun,
      x0=np.log(y0),
      jac=self.full_system_jac,
      bounds=(bound_min, bound_max),
      **self.lsq_opts
    ).x
    return np.exp(x)

  def _full_system_fun(self, y: torch.Tensor) -> torch.Tensor:
    return self.system._fun(t=0.0, y=torch.exp(y))

  # Conservative variables
  # ===================================
  def from_cons(
    self,
    rho: float,
    e: float
  ) -> np.ndarray:
    """Compute equilibirum state from conservative macrosocpiv variables,
    such as density and total energy.
    """
    # Convert to 'torch.Tensor'
    rho, e = [bkd.to_torch(z).reshape(1) for z in (rho, e)]
    # Update mixture
    self.system.mix.set_rho(rho)
    # Compute electron molar fraction and temperaure
    x = sp.optimize.least_squares(
      fun=self.from_cons_fun,
      x0=np.log([1e-1,1e4]),
      jac=self.from_cons_jac,
      bounds=([-np.inf, -np.inf], [0.0, np.log(1e5)]),
      args=(e,),
      **self.lsq_opts
    ).x
    # Extract variables
    x_em, T = [z.reshape(1) for z in bkd.to_torch(np.exp(x))]
    x_em = self._clipping(x_em)
    # Update species thermo
    self.system.mix.update_species_thermo(T)
    # Update composition
    self._update_composition(x_em)
    # Compute number densities
    n = self.system.mix.get_qoi_vec("n")
    y = torch.cat([n, T, T])
    return bkd.to_numpy(y)

  def _from_cons_fun(
    self,
    x: torch.Tensor,
    e: torch.Tensor
  ) -> torch.Tensor:
    # Extract variables
    x_em, T = torch.exp(x)
    # Update species thermo
    self.system.mix.update_species_thermo(T)
    # Update composition
    self._update_composition(x_em)
    # Update mixture thermo
    self.system.mix.update_mixture_thermo()
    # Enforce detailed balance
    f0 = self._detailed_balance()
    # Enforce consrvation of energy
    f1 = self.system.mix.e / e - 1.0
    return torch.cat([f0,f1])

  # Utils
  # ===================================
  def _update_composition(self, x_em: torch.Tensor) -> None:
    """Set numer densities given electron molar fraction using conservation
    of charges (eq 1) and mass (eq 2)"""
    x = torch.zeros(self.system.mix.nb_comp)
    # Electron
    s = self.system.mix.species["em"]
    x[s.indices] = x_em
    # Argon ion
    s = self.system.mix.species["Arp"]
    x[s.indices] = x_em * s.q / s.Q
    # Argon neutral
    s = self.system.mix.species["Ar"]
    x[s.indices] = (1.0-2.0*x_em) * s.q / s.Q
    # Update composition
    self.system.mix.update_composition_x(x)

  def _detailed_balance(self) -> torch.Tensor:
    n, Q = [self._get_species_attr(k) for k in ("n", "Q")]
    l = torch.sum(n["Arp"]) * n["em"] / torch.sum(n["Ar"])
    r = Q["Arp"] * Q["em"] / Q["Ar"]
    f = l/r - 1.0
    return f.reshape(1)

  def _get_species_attr(self, attr: str) -> Dict[str, torch.Tensor]:
    return {k: getattr(s, attr) for (k, s) in self.system.mix.species.items()}

  def _clipping(self, x: torch.Tensor) -> torch.Tensor:
    if self.clipping:
      return torch.clip(x, const.XMIN, 1.0)
    else:
      return x

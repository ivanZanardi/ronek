import time
import torch
import numpy as np
import scipy as sp

from .. import const
from .. import utils
from .. import backend as bkd
from .sources import Sources
from .mixture import Mixture
from .kinetics import Kinetics
from typing import Dict, Union, Optional


class Equilibrium(object):

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
    mixture: Mixture,
    clipping: bool = True
  ) -> None:
    self.mix = mixture
    self.clipping = clipping

  # Primitive variables
  # ===================================
  def from_prim(
    self,
    rho: float,
    T: float,
    Te: Optional[float] = None
  ) -> np.ndarray:
    """Compute equilibirum state from primritive macrosocpiv variables,
    such as density temperarue and electron temperatue.
    """
    # Convert to 'torch.Tensor'
    rho, T, Te = [bkd.to_torch(z).reshape(1) for z in (rho, T, Te)]
    # Update mixture
    self.mix.set_rho(rho)
    self.mix.update_species_thermo(T, Te)
    # Compute electron molar fraction
    x = sp.optimize.leastsq(
      func=self._from_prim_fun_np,
      x0=np.log(0.1),
      Dfun=self._from_prim_jac_np,
      maxfev=int(1e5)
    )[0]
    x = bkd.to_torch(np.exp(x))
    if self.clipping:
      x = torch.clip(x, const.XMIN, 1.0)
    # Update composition
    self._update_composition(x)
    # Return number densities
    n = self.mix.get_qoi_vec("n")
    return bkd.to_numpy(n)

  def _from_prim_fun_np(self, x: np.ndarray) -> np.ndarray:
    x = bkd.to_torch(x)
    f = self._from_prim_fun(x)
    return bkd.to_numpy(f)

  def _from_prim_jac_np(self, x: np.ndarray) -> np.ndarray:
    x = bkd.to_torch(x)
    j = torch.func.jacrev(self._from_prim_fun)(x)
    return bkd.to_numpy(j)

  def _from_prim_fun(self, x: torch.Tensor) -> torch.Tensor:
    # Electron molar fraction
    x_em = torch.exp(x)
    # Update composition
    self._update_composition(x_em)
    # Enforce detailed balance
    return self._detailed_balance()

  def _update_composition(self, x_em: torch.Tensor) -> None:
    """Set numer densities given electron molar fraction using conservation
    of charges (eq 1) and mass (eq 2)"""
    x = torch.zeros(self.mix.nb_comp)
    # Electron
    s = self.mix.species["em"]
    x[s.indices] = x_em
    # Argon ion
    s = self.mix.species["Arp"]
    x[s.indices] = x_em * s.q / s.Q
    # Argon neutral
    s = self.mix.species["Ar"]
    x[s.indices] = (1.0-2.0*x_em) * s.q / s.Q
    # Update composition
    self.mix.update_composition_x(x)

  def _detailed_balance(self) -> torch.Tensor:
    n, Q = [self._get_species_attr(k) for k in ("n", "Q")]
    l = torch.sum(n["Arp"]) * n["em"] / torch.sum(n["Ar"])
    r = Q["Arp"] * Q["em"] / Q["Ar"]
    f = l/r - 1.0
    return f.reshape(1)

  def _get_species_attr(self, attr: str) -> Dict[str, torch.Tensor]:
    return {k: getattr(s, attr) for (k, s) in self.mix.species.items()}

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
    self.mix.set_rho(rho)
    # Compute electron molar fraction
    x0 = np.array([1e-1,1e4])
    x = sp.optimize.leastsq(
      func=self._from_cons_fun_np,
      x0=np.log(x0),
      args=(e,),
      Dfun=self._from_cons_jac_np,
      maxfev=int(1e5)
    )[0]
    # Extract variables
    x_em, T = bkd.to_torch(np.exp(x))
    if self.clipping:
      x_em = torch.clip(x_em, const.XMIN, 1.0)
    # Update species thermo
    self.mix.update_species_thermo(T)
    # Update composition
    self._update_composition(x_em)
    # Return number densities
    n = self.mix.get_qoi_vec("n")
    return bkd.to_numpy(n)

  def _from_cons_fun(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    # Extract variables
    x_em, T = torch.exp(x)
    # Update species thermo
    self.mix.update_species_thermo(T)
    # Update composition
    self._update_composition(x_em)
    # Update mixture thermo
    self.mix.update_mixture_thermo()
    # Enforce detailed balance
    f0 = self._detailed_balance()
    # Enforce consrvation of energy
    f1 = self.mix.e / e - 1.0
    return torch.cat([f0,f1])

  def _from_cons_fun_np(self, x: np.ndarray, e) -> np.ndarray:
    x = bkd.to_torch(x)
    f = self._from_cons_fun(x, e)
    return bkd.to_numpy(f)

  def _from_cons_jac_np(self, x: np.ndarray, e) -> np.ndarray:
    x = bkd.to_torch(x)
    j = torch.func.jacrev(self._from_cons_fun)(x, e)
    return bkd.to_numpy(j)

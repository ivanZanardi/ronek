import numpy as np

from .. import const
from species import Species
from typing import Dict, Union


class Mixture(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    species: Dict[str, Union[str, dict]],
    species_order: tuple = (),
    use_factorial: bool = True
  ):
    self.species_order = tuple(species_order)
    self.use_factorial = bool(use_factorial)
    self._init_species(species)

  def _init_species(self, species: Dict[str, str]) -> None:
    # Initialize species
    self.species = {}
    for k in species.keys():
      self.species[k] = Species(species[k], self.use_factorial)
    # Index species
    si = 0
    for k in self.species_order:
      ei = si + self.species[k].nb_comp
      self.species[k].indices = np.arange(si, ei)
      si = ei
    self.nb_comp = ei

  # Build
  # ===================================
  def build(self) -> None:
    for s in self.species.values():
      s.build()
    self._build_mass_matrix()

  def _build_mass_matrix(self) -> None:
    # Build vector of masses
    self.m = np.ones(self.nb_comp)
    for k in self.species_order:
      s = self.species[k]
      self.m[s.indices] *= s.m
    self.m_inv = 1.0/self.m
    # Build mass matrix
    self.m_mat = np.diag(self.m)
    self.m_inv_mat = np.diag(self.m_inv)

  # Update
  # ===================================
  def update(self, w, rho, T, Te=None) -> None:
    # Update species thermo
    self._update_species_thermo(T, Te)
    # Update composition
    self._update_composition(w, rho)
    # Update mixture thermo
    self._update_mix_thermo()

  def _update_species_thermo(self, T, Te=None) -> None:
    if (Te is None):
      Te = T
    for s in self.species.values():
      Ti = Te if (s.name == "e-") else T
      s.update(Ti)

  def _update_composition(self, w, rho) -> None:
    # Set species mass fractions
    for s in self.species.values():
      s.w = w[s.indices]
      s.rho = s.w * rho
    # Mixture molar mass [kg/mol]
    self._M("w")
    # Mixture specific gas constants [J/(kg K)]
    self._R()
    # Set species molar fractions
    self._convert_mass_mole("w")

  def _update_mix_thermo(self) -> None:
    # Constant specific heats
    self.cv_h = self._cv_h()
    # Energies
    self.e = self._e()
    self.e_e = self._e_e()
    self.e_int_h = self._e_int_h()
    self.hf_h = self._hf_h()

  # Conversions
  # ===================================
  def get_w(
    self,
    n: np.ndarray,
    rho: float
  ) -> np.ndarray:
    return (1/rho) * self.m_mat @ n

  def get_n(
    self,
    w: np.ndarray,
    rho: float
  ) -> np.ndarray:
    return rho * self.m_inv_mat @ w

  def get_rho(
    self,
    n: np.ndarray
  ) -> float:
    return np.diag(self.m_mat) @ n

  # Equilibrium composition
  # ===================================
  def compute_eq_comp(
    self,
    p: float,
    T: float
  ) -> np.ndarray:
    # Solve this system of equations:
    # -------------
    # 1) Charge neutrality:
    #    x_em = x_Arp
    # 2) Mole conservation:
    #    x_em + x_Arp + x_Ar = 1
    # 3) Detailed balance:
    #    (n_Arp*n_em)/n_Ar = (Q_Arp*Q_em)/Q_Ar
    # -------------
    # Update thermo
    self._update_species_thermo(T)
    # Compute number density
    n = p / (const.UKB*T)
    # Compute coefficient for quadratic system
    f = self.species["Ar+"].Q * self.species["e-"].Q
    f /= (self.species["Ar"].Q * n)
    # Solve quadratic system for 'x_em'
    a = 1.0
    b = 2.0 * f
    c = -f
    x = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    # Set molar fractions
    s = self.species["e-"]
    s.x = x
    s = self.species["Ar+"]
    s.x = x * s.q / s.Q
    s = self.species["Ar"]
    s.x = (1.0-2.0*x) * s.q / s.Q
    # Set mass fractions
    self._M("x")
    self._set_w_s()
    # Number densities
    n = n * np.concatenate([self.species[k].x for k in self.species_order])
    # Mass densities
    return self.get_rho(n)

  # Initial composition
  # ===================================
  def get_init_composition(
    self,
    p: float,
    T: float,
    noise: bool = False,
    sigma: float = 1e-2
  ) -> np.ndarray:
    # Compute equilibrium mass densities
    rho = self.compute_eq_comp(p, T)
    # Add random noise
    if noise:
      # > Electron
      s = self.species["e-"]
      x_em = np.clip(s.x * self._add_norm_noise(s, sigma), 0, 1)
      s.x = x_em
      # > Argon Ion
      s = self.species["Ar+"]
      s.x = x_em * self._add_norm_noise(s, sigma)
      # > Argon
      s = self.species["Ar"]
      s.x = (1.0-2.0*x_em) * self._add_norm_noise(s, sigma)
      # Set mass fractions
      self._M("x")
      self._set_w_s()
      # Mass densities
      w = w * np.concatenate([self.species[k].w for k in self.species_order])
      rho = w * np.sum(rho)
    # Return mass densities
    return rho

  def _add_norm_noise(self, species, sigma):
    f = 1.0 + sigma*np.random.rand(species.nb_comp)
    q = species.q * f
    return q / np.sum(q)

  # Mixture properties
  # ===================================
  def _M(
    self,
    qoi_used: str = "w"
  ):
    self.M = 0.0
    if (qoi_used == "w"):
      for s in self.species.values():
        self.M += np.sum(s.w) / s.M
      self.M = 1.0/self.M
    elif (qoi_used == "x"):
      for s in self.species.values():
        self.M += np.sum(s.x) * s.M

  def _R(self):
    self.R = 0.0
    for s in self.species.values():
      self.R += np.sum(s.w) * s.R

  def _convert_mass_mole(
    self,
    qoi_used: str = "w"
  ):
    if (qoi_used == "w"):
      self._set_x_s()
    elif (qoi_used == "x"):
      self._set_w_s()

  def _set_x_s(self):
    for s in self.species.values():
      s.x = self.M / s.M * s.w

  def _set_w_s(self):
    for s in self.species.values():
      s.w = s.M / self.M * s.x

  # Constant specific heats
  # -----------------------------------
  def _cv_h(self):
    # Heavy particle constant-volume specific heat [J/(kg K)]
    cv_h = 0.0
    for s in self.species.values():
      if (s.name != "e-"):
        cv_h += np.sum(s.w * s.cv)
    return cv_h

  # Energies
  # -----------------------------------
  def _e(self):
    # Total energy [J/kg]
    e = 0.0
    for s in self.species.values():
      e += np.sum(s.w * s.e)
    return e

  def _e_e(self):
    # Electron energy [J/kg]
    s = self.species["e-"]
    return np.sum(s.w * s.e)

  def _e_int_h(self):
    # Heavy particle internal energy [J/kg]
    e_int_h = 0.0
    for s in self.species.values():
      if (s.name != "e-"):
        e_int_h += np.sum(s.w * s.e_int)
    return e_int_h

  def _hf_h(self):
    # Heavy particle enthalpy of formation [J/kg]
    hf_h = 0.0
    for s in self.species.values():
      if (s.name != "e-"):
        hf_h += np.sum(s.w * s.hf)
    return hf_h

  # Temperatures
  # -----------------------------------
  def compute_temp(
    self,
    e: float,
    e_e: float
  ):
    # Heavy particle temperature [K]
    T = (e - e_e - self.e_int_h - self.hf_h) / self.cv_h
    # Free electron temperature [K]
    s = self.species["e-"]
    Te = e_e / (s.w * s.cv)
    # Clipping
    T = np.maximum(T, const.TMIN)
    Te = np.maximum(Te, const.TMIN)
    return T, Te

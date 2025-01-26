import torch
import numpy as np
import scipy as sp

from ... import const
from ... import backend as bkd
from .mixture import Mixture
from .species import Species
from typing import Dict


class Equilibrium(object):

  """
  Class to compute the equilibrium state (mass fractions and temperature)
  of a system involving argon and its ionized species (Ar, Ar+, and e^-).

  The equilibrium state is determined by solving the following system of
  equations:
  1) **Charge neutrality**:
    \[
    x_{e^-} = x_{\text{Ar}^+}
    \]
    where \( x_{e^-} \) is the electron molar fraction and
    \( x_{\text{Ar}^+} \) is the argon ion molar fraction.
  2) **Mole conservation**:
    \[
    x_{e^-} + x_{\text{Ar}^+} + x_{\text{Ar}} = 1
    \]
    where \( x_{\text{Ar}} \) is the molar fraction of neutral argon.
  3) **Detailed balance**:
    \[
    \frac{n_{\text{Ar}^+} n_{e^-}}{n_{\text{Ar}}} =
    \frac{Q_{\text{Ar}^+} Q_{e^-}}{Q_{\text{Ar}}}
    \]
    This describes the ionization equilibrium between neutral argon, argon
    ions, and electrons in the system, where \( n \) represents the number
    density and \( Q \) represents the charge.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    mixture: Mixture,
    clipping: bool = True
  ) -> None:
    """
    Initializes the equilibrium solver for a given mixture.

    :param mixture: The `Mixture` object representing the chemical mixture.
    :type mixture: `Mixture`
    :param clipping: Flag to control whether molar fractions are
                     clipped to avoid too small values.
    :type clipping: bool, optional, default is True
    """
    self.mix = mixture
    self.clipping = clipping
    self.lsq_opts = dict(
      method="trf",
      ftol=bkd.epsilon(),
      xtol=bkd.epsilon(),
      gtol=0.0,
      max_nfev=int(1e5)
    )
    self.set_fun_jac()

  def set_fun_jac(self) -> None:
    """
    Set up functions and their Jacobians for the least-squares optimization
    for both the primitive (`from_prim`) and conservative (`from_cons`)
    variable formulations.
    """
    for name in ("from_prim", "from_cons"):
      # Function
      fun = getattr(self, f"_{name}_fun")
      setattr(self, f"{name}_fun", bkd.make_fun_np(fun))
      # Jacobian
      jac = torch.func.jacrev(fun, argnums=0)
      setattr(self, f"{name}_jac", bkd.make_fun_np(jac))

  # Initial solution
  # ===================================
  def get_init_sol(
    self,
    mu: np.ndarray,
    noise: bool = False,
    sigma: float = 1e-2
  ) -> np.ndarray:
    """
    Generates an initial solution for heat bath simulations based on the input
    parameters `mu`.

    This method computes the equilibrium state of the system, including partial
    densities, translational temperature, and electron pressure. It optionally
    adds random noise to the equilibrium state and replaces the equilibrium
    initial electron temperature with the translational temperature for heat
    bath simulations.

    :param mu: A NumPy array containing density, translational temperature,
               and electron temperature.
    :type mu: np.ndarray
    :param noise: Flag indicating whether to add random noise to the
                  equilibrium state.
    :type noise: bool, optional, default is False
    :param sigma: Standard deviation for the noise (if `noise` is True).
    :type sigma: float, optional, default is 1e-2

    :return: The equilibrium state vector, including partial densities,
             translational temperature, and electron pressure.
    :rtype: np.ndarray
    """
    # Unpack the input array into individual parameters
    rho, T, Te = bkd.to_torch(mu)
    # Compute the equilibrium state based on rho and Te
    y = self.from_prim(rho=rho, T=Te)
    # If noise requested, update the composition and recompute the state vector
    if noise:
      self._update_composition(
        x_em=self.mix.species["em"].x,
        noise=noise,
        sigma=sigma
      )
      y = self._compose_state_vector(
        rho=rho,
        T=Te,
        x_em=self.mix.species["em"].x
      )
    # Replace the equilibrium temperature Te with T for heat bath simulation
    y[-2] = T
    return bkd.to_numpy(y)

  # Primitive variables
  # ===================================
  def from_prim(
    self,
    rho: float,
    T: float
  ) -> np.ndarray:
    """
    Compute the equilibrium state from primitive macroscopic variables
    such as density and temperature.

    :param rho: Density of the system.
    :type rho: float
    :param T: Temperature of the system.
    :type T: float

    :return: The equilibrium state vector, including partial densities,
             translational temperature, and electron pressure.
    :rtype: np.ndarray
    """
    # Convert to 'torch.Tensor'
    rho, T = [bkd.to_torch(z).reshape(1) for z in (rho, T)]
    # Update mixture
    self.mix.set_rho(rho)
    self.mix.update_species_thermo(T)
    # Compute electron molar fraction
    x = sp.optimize.least_squares(
      fun=self.from_prim_fun,
      x0=np.log([1e-2]),
      jac=self.from_prim_jac,
      bounds=(-np.inf, 0.0),
      **self.lsq_opts
    ).x
    # Extract variables
    x_em = bkd.to_torch(np.exp(x))
    x_em = self._clipping(x_em)
    # Compose state vector
    y = self._compose_state_vector(rho, T, x_em)
    return bkd.to_numpy(y)

  def _from_prim_fun(self, x: torch.Tensor) -> torch.Tensor:
    """
    Compute the residuals of the detailed balance condition based on the
    electron molar fraction.

    This method takes the electron molar fraction (as the logarithm of
    the fraction), updates the species composition based on the value,
    and enforces the detailed balance condition for the reaction:
    \[
    \text{Ar} \rightleftharpoons \text{Ar}^+ + e^-
    \]

    :param x: Logarithm of the electron molar fraction.
    :type x: torch.Tensor

    :return: The residuals of the detailed balance condition.
    :rtype: torch.Tensor
    """
    # Extract variables
    x_em = torch.exp(x)
    # Update composition
    self._update_composition(x_em)
    # Enforce detailed balance
    return self._detailed_balance()

  # Conservative variables
  # ===================================
  def from_cons(
    self,
    rho: float,
    e: float
  ) -> np.ndarray:
    """
    Computes the equilibrium state from conservative macroscopic variables
    such as density and total energy.

    It determines the electron molar fraction (\(x_{\text{e}^-}\)) and
    temperature (\(T\)) that satisfy the conservation equations and
    detailed balance conditions.

    :param rho: Mass density of the system.
    :type rho: float
    :param e: Total energy of the system (per unit volume).
    :type e: float

    :return: The equilibrium state vector, including partial densities,
             translational temperature, and electron pressure.
    :rtype: np.ndarray
    """
    # Convert to 'torch.Tensor'
    rho, e = [bkd.to_torch(z) for z in (rho, e)]
    # Update mixture
    self.mix.set_rho(rho)
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
    self.mix.update_species_thermo(T)
    # Compose state vector
    y = self._compose_state_vector(rho, T, x_em)
    return bkd.to_numpy(y)

  def _from_cons_fun(
    self,
    x: torch.Tensor,
    e: torch.Tensor
  ) -> torch.Tensor:
    """
    This method compute the residuals of two key constraints:
    1. **Detailed Balance**: Ensures equilibrium for the reaction
       \(\text{Ar} \leftrightarrow \text{Ar}^+ + e^-\).
    2. **Energy Conservation**: Ensures the total energy of the system matches
       the specified input energy `e`.

    :param x: Tensor containing logarithmic values of the electron molar
              fraction and temperature \([ \ln(x_{\text{e}^-}), \ln(T) ]\).
    :type x: torch.Tensor
    :param e: Total energy of the system (per unit volume).
    :type e: torch.Tensor

    :return: Tensor of residuals from the conservation equations. The first
             component corresponds to the detailed balance equation, and the
             second component to energy conservation.
    :rtype: torch.Tensor
    """
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
    # Enforce conservation of energy
    f1 = self.mix.e / e - 1.0
    return torch.cat([f0,f1])

  # Utils
  # ===================================
  def _update_composition(
    self,
    x_em: torch.Tensor,
    noise: bool = False,
    sigma: float = 1e-2
  ) -> None:
    """
    Updates the species composition based on the electron molar fraction
    and optionally adds random noise to the composition.

    This method modifies the composition vector using the electron molar
    fraction `x_em`, applying noise if specified, and updates the species
    states for various elements like "Ar" and "Arp".

    :param x_em: Electron molar fraction of the species.
    :type x_em: torch.Tensor
    :param noise: Whether to add random noise to the composition.
    :type noise: bool, optional, default is False
    :param sigma: Standard deviation of the noise if `noise` is True.
    :type sigma: float, optional, default is 1e-2
    """
    x = torch.zeros(self.mix.nb_comp)
    # Electron
    s = self.mix.species["em"]
    if noise:
      x_em *= self._add_norm_noise(s, sigma, use_q=False)
    x_em = self._clipping(x_em)
    x[s.indices] = x_em
    # Argon neutral/ion
    for k in ("Ar", "Arp"):
      s = self.mix.species[k]
      x = x_em if (k == "Arp") else 1.0-2.0*x_em
      f = self._add_norm_noise(s, sigma) if noise else s.q / s.Q
      x[s.indices] = x * f
    # Update composition
    self.mix.update_composition_x(x)

  def _add_norm_noise(
    self,
    species: Species,
    sigma: float = 1e-2,
    use_q: bool = True
  ) -> torch.Tensor:
    """
    Adds unit norm random noise to the species composition.

    The noise is generated according to the specified standard deviation
    `sigma`, and optionally scaled by the partition function `q` for the
    species.

    :param species: The species for which the noise is applied.
    :type species: `Species`
    :param sigma: Standard deviation of the noise to be added.
    :type sigma: float
    :param use_q: Whether to scale the noise by the partition function `q`.
    :type use_q: bool, optional, default is True

    :return: A tensor containing the noisy composition values.
    :rtype: torch.Tensor
    """
    f = 1.0 + sigma * torch.rand(species.nb_comp)
    if use_q:
      f *= species.q
      f /= torch.sum(f)
    return f

  def _detailed_balance(self) -> torch.Tensor:
    r"""
    This method calculates the detailed balance error for the reaction:
    \[
    \text{Ar} \leftrightarrow \text{Ar}^+ + e^-
    \]

    :return: A tensor representing the deviation from detailed balance.
             A value close to zero indicates equilibrium.
    :rtype: torch.Tensor
    """
    n, Q = [self._get_species_attr(k) for k in ("n", "Q")]
    l = torch.sum(n["Arp"]) * n["em"] / torch.sum(n["Ar"])
    r = Q["Arp"] * Q["em"] / Q["Ar"]
    f = l/r - 1.0
    return f.reshape(1)

  def _get_species_attr(self, attr: str) -> Dict[str, torch.Tensor]:
    """
    Retrieve the specified attribute for all species in the mixture.

    This method fetches the requested attribute (e.g., number density,
    partition function) for each species in the mixture and returns a
    dictionary mapping species names to their corresponding attribute values.

    :param attr: The name of the attribute to retrieve (e.g., "n" for
                 number density, "Q" for partition function).
    :type attr: str

    :return: A dictionary where the keys are species names and the values are
             the corresponding attribute tensors.
    :rtype: Dict[str, torch.Tensor]
    """
    return {k: getattr(s, attr) for (k, s) in self.mix.species.items()}

  def _clipping(self, x: torch.Tensor) -> torch.Tensor:
    """
    Clip molar fractions to avoid values that are too small, ensuring
    numerical stability.

    This method ensures that the molar fractions remain within a valid range.
    If clipping is enabled, the values are constrained between a predefined
    minimum (`const.XMIN`) and a maximum of 1.0. If clipping is disabled,
    the input tensor is returned unchanged.

    :param x: Tensor of molar fractions to be clipped.
    :type x: torch.Tensor

    :return: Tensor of molar fractions after applying clipping (if enabled).
    :rtype: torch.Tensor
    """
    if self.clipping:
      return torch.clip(x, const.XMIN, 1.0)
    else:
      return x

  def _compose_state_vector(
    self,
    rho: torch.Tensor,
    T: torch.Tensor,
    x_em: torch.Tensor
  ) -> torch.Tensor:
    """
    Composes the state vector for the system's equilibrium state.

    This method combines partial densities, translational temperature,
    and electron pressure into a single vector. It updates the species
    composition based on the electron molar fraction and computes
    electron pressure and other quantities.

    :param rho: Density of the system.
    :type rho: torch.Tensor
    :param T: Translational temperature of the system.
    :type T: torch.Tensor
    :param x_em: Electron molar fraction of the species.
    :type x_em: torch.Tensor

    :return: Equilibrium state vector, including partial densities,
             translational temperature, and electron pressure.
    :rtype: torch.Tensor
    """
    self._update_composition(x_em)
    pe = self.mix.get_pe(Te=T, ne=self.mix.species["em"].n)
    w = self.mix.get_qoi_vec("w")
    return torch.cat([rho*w, T, pe])

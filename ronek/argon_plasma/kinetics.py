import copy
import numpy as np
import scipy as sp
import dill as pickle

from .. import const
from . import chem_eq


class Kinetics(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    mixture,
    reactions,
    use_Q11_fit=False
  ):
    # Set mixtures
    self.mix = mixture                  # Reference mixture
    self.mix_e = copy.deepcopy(mixture) # Electron temperature-based thermo
    # Load reactions rates
    self._init_reactions(reactions)
    # Q11 collision integrals
    self.use_Q11_fit = use_Q11_fit

  def _init_reactions(self, reactions):
    self.reactions = reactions
    if (not isinstance(self.reactions, dict)):
      self.reactions = pickle.load(open(self.reactions, "rb"))
    for (name, reaction) in self.reactions.items():
      if (name not in ("T", "EN", "EI")):
        self.reactions[name]["param"] = chem_eq.get_param(reaction["equation"])
    # Electron-neutral collision rate (EN)
    if ("EN" in self.reactions):
      x = self.reactions["T"].reshape(-1)
      y = self.reactions["EN"]["values"].reshape(-1)
      self.reactions["EN"]["interp"] = sp.interpolate.interp1d(x, y, "linear")

  # Rates
  # ===================================
  def update(self, T, Te):
    # Update electron temperature-based thermo
    self.mix_e.update_species_thermo(Te)
    # Compute reaction rates
    # > Zeroth order moment
    self.rates = {}
    if ("EXh" in self.reactions):
      self.rates["EXh"] = self._compute_EXh_rates(T)
    if ("EXe" in self.reactions):
      self.rates["EXe"] = self._compute_EXe_rates(Te)
    if ("Ih" in self.reactions):
      self.rates["Ih"] = self._compute_Ih_rates(T)
    if ("Ie" in self.reactions):
      self.rates["Ie"] = self._compute_Ie_rates(Te)
    # > First order moment
    if ("EN" in self.reactions):
      self.rates["EN"] = self._compute_EN_rate(Te)
      self.rates["EI"] = self._compute_EI_rate(T, Te)

  # Forward and backward rates
  # -----------------------------------
  def _compute_fwd_rates(self, T, A, beta, Ta):
    # Arrhenius law
    return A * np.exp(beta*np.log(T) - Ta/T)

  def _compute_bwd_rates(self, reaction, mixture, kf):
    # Initialize backward rates
    kb = copy.deepcopy(kf)
    # Set shapes
    k_shape = kb.shape
    nb_species = len(k_shape)
    q_shape = tuple([1]*nb_species)
    # Loov over species
    i = 0
    for side in ("reactants", "products"):
      for (stoich, name, level) in reaction["param"][side]:
        # Define i-th partition function
        qi = mixture.species[name].q
        if (level != "*"):
          qi = qi[level]
        if (stoich > 1):
          qi = qi**stoich
        if (side == "products"):
          qi = 1.0/qi
        # Reshape i-th partition function
        qi_shape = list(q_shape)
        qi_shape[i] = k_shape[i]
        qi = qi.reshape(qi_shape)
        # Apply i-th partition function
        kb *= qi
        # Update species counter
        i += 1
    # Transpose backward rates
    axes = np.arange(nb_species)
    shift = -reaction["param"]["nb_reactants"]
    return np.transpose(kb, axes=np.roll(axes, shift=shift))

  # Collisional processes - Zeroth order moment
  # -----------------------------------
  def _compute_EXh_rates(self, T, identifier="EXh"):
    """
    Excitation by heavy-particle impact (EXh)
    - Equation:       Ar(*)+Ar(0)=Ar(*)+Ar(0)
    - Forward rate:   kf = kf(T)
    - Backward rate:  kb = kb(T)
    """
    reaction = self.reactions[identifier]
    kf = self._compute_fwd_rates(T, **reaction["values"])
    kb = self._compute_bwd_rates(reaction, self.mix, kf)
    return {"fwd": kf, "bwd": kb}

  def _compute_EXe_rates(self, Te, identifier="EXe"):
    """
    Excitation by electron impact (EXe)
    - Equation:       Ar(*)+em=Ar(*)+em
    - Forward rate:   kf = kf(Te)
    - Backward rate:  kb = kb(Te)
    """
    reaction = self.reactions[identifier]
    kf = self._compute_fwd_rates(Te, **reaction["values"])
    kb = self._compute_bwd_rates(reaction, self.mix_e, kf)
    return {"fwd": kf, "bwd": kb}

  def _compute_Ih_rates(self, T, identifier="Ih"):
    """
    Ionization by heavy-particle impact (Ih)
    - Equation:       Ar(*)+Ar(0)=Arp(*)+em+Ar(0)
    - Forward rate:   kf = kf(T)
    - Backward rate:  kb = kb(T,Te)
    """
    reaction = self.reactions[identifier]
    kf = self._compute_fwd_rates(T, **reaction["values"])
    kb = self._compute_bwd_rates(reaction, self.mix, kf)
    return {"fwd": kf, "bwd": kb}

  def _compute_Ie_rates(self, Te, identifier="Ie"):
    """
    Ionization by electron impact (Ie)
    - Equation:       Ar(*)+em=Arp(*)+em+em
    - Forward rate:   kf = kf(Te)
    - Backward rate:  kb = kb(Te)
    """
    reaction = self.reactions[identifier]
    kf = self._compute_fwd_rates(Te, **reaction["values"])
    kb = self._compute_bwd_rates(reaction, self.mix_e, kf)
    return {"fwd": kf, "bwd": kb}

  # Collisional processes - First order moment
  # -----------------------------------
  def _compute_EN_rate(self, Te, identifier="EN"):
    """Electron-neutral collision rate/integral (EN)"""
    if self.use_Q11_fit:
      # Curve fit model (Gupta)
      # > Common factors
      lnT1 = np.log(Te)
      lnT2 = lnT1*lnT1
      lnT3 = lnT2*lnT1
      # > Collision integral for 'em-Ar' interactions
      c = self.reactions[identifier]["Q11_fit"]
      Q11 = 1e-20 * np.exp(c[0]*lnT3 + c[1]*lnT2 + c[2]*lnT1 + c[3])
      return Q11
    else:
      # Look-up table (Kapper's PhD thesis, 2009)
      return self.reactions["EN"]["interp"](Te)

  def _compute_EI_rate(self, T, Te, identifier="EI"):
    """Electron-ion collision rate/integral (EI)"""
    # Electron species
    s = self.mix.species["em"]
    if self.use_Q11_fit:
      # Curve fit model (Magin's PhD thesis, ULB, 2004)
      # > Average closest impact parameter for 'em-Arp' and 'em-em' interactions
      f0 = const.UE**2/(8.0*np.pi*const.UEPS0*const.UKB)
      be = f0/Te
      # > Average closest impact parameter for 'Arp-Arp' interactions
      bh = f0/T
      # > Debye shielding distance (em and Arp contributions)
      Ds = np.sqrt(const.UEPS0*const.UKB*Te/(2.0*s.n*const.UE**2))
      Ds = np.minimum(Ds,10000.0*(be + bh))
      # > Non-dimensional temperature for charge-charge interactions
      Tse = np.maximum(Ds/(2.0*be),0.1)
      # > Common factors
      lnT1 = np.log(Tse)
      lnT2 = lnT1*lnT1
      lnT3 = lnT2*lnT1
      lnT4 = lnT3*lnT1
      # > Collision integral for 'em-Arp' and 'em-em' interactions
      c = self.reactions[identifier]["Q11_fit"]
      Q11 = np.exp(c[0]*lnT4 + c[1]*lnT3 + c[2]*lnT2 + c[3]*lnT1 + c[4])
      Q11 *= np.pi*Ds**2/(Tse**2)
      return Q11
    else:
      # Analytical model (Kapper's PhD thesis, 2009)
      # > Compute Coulomb logarithm
      lam = 1.24e7*np.sqrt(Te**3/s.n)
      # > Compute momentum-averaged cross section
      Qei = 5.85e-10*np.log(lam)/Te**2
      # > Electron mean thermal velocity
      ve = np.sqrt((8.0*const.URG*Te)/(np.pi*s.M))
      return ve*Qei

import copy
import torch
import numpy as np
import dill as pickle

from .. import chem_eq
from ... import const
from ... import utils
from ... import backend as bkd
from pyharm import PolyHarmInterpolator


class Kinetics(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    mixture,
    reactions,
    use_fit=False
  ):
    # Set mixtures
    self.mix = mixture                  # Reference mixture
    self.mix_e = copy.deepcopy(mixture) # Electron temperature-based thermo mixture
    # Collision integrals fit
    self.use_fit = use_fit
    # Initialize reactions rates
    self._init_reactions(reactions)

  def _init_reactions(self, reactions):
    # Load reactions
    self.reactions = reactions
    if (not isinstance(self.reactions, dict)):
      self.reactions = pickle.load(open(self.reactions, "rb"))
    # Convert reactions
    self.reactions = utils.map_nested_dict(self.reactions, bkd.to_torch)
    # Get reaction parameters
    for (name, reaction) in self.reactions.items():
      if (name not in ("T", "EN", "EI")):
        self.reactions[name]["param"] = chem_eq.get_param(reaction["equation"])
    self._init_en_rate()

  def _init_en_rate(self):
    # Electron-neutral collision rate (EN)
    if (("EN" in self.reactions) and (not self.use_fit)):
      self.reactions["EN"]["interp"] = PolyHarmInterpolator(
        c=self.reactions["T"].reshape(1,-1,1),
        f=self.reactions["EN"]["values"].reshape(1,-1,1),
        order=1,
        smoothing=0.0,
        dtype=bkd.floatx(bkd="torch")
      )

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
      ve = self._compute_ve(Te)
      self.rates["EN"] = self._compute_en_rate(Te, ve)
      self.rates["EI"] = self._compute_ei_rate(T, Te, ve)
    # Squeeze tensors
    self.rates = utils.map_nested_dict(self.rates, torch.squeeze)

  def _compute_ve(self, Te):
    """Electron mean thermal velocity"""
    return torch.sqrt((8.0*const.UKB*Te)/(torch.pi*const.UME))

  # Forward and backward rates
  # -----------------------------------
  def _compute_fwd_rates(self, T, A, beta, Ta):
    # Arrhenius law
    return A * torch.exp(beta*torch.log(T) - Ta/T)

  def _compute_bwd_rates(self, reaction, mixture, kf):
    # Initialize backward rates
    kb = torch.clone(kf)
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
    shifts = -reaction["param"]["nb_reactants"]
    dims = np.arange(nb_species)
    dims = np.roll(dims, shift=shifts).tolist()
    return torch.permute(kb, dims=dims)

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
  def _compute_en_rate(self, Te, ve):
    """Electron-neutral collision rate (EN)"""
    if self.use_fit:
      # Curve fit model
      # > See: https://doi.org/10.1007/978-1-4419-8172-1 - Eq. 11.3
      return 8.0/3.0 * ve * self._compute_en_Q11_capitelli(Te)
    else:
      # Look-up table
      # > See: Kapper's PhD thesis, The Ohio State University, 2009
      return 2.0 * self.reactions["EN"]["interp"](Te.reshape(1,1,1)).squeeze()

  def _compute_en_Q11_capitelli(self, Te):
    c = self.reactions["EN"]["Q11_fit"]
    lnT = torch.log(Te)
    fac = torch.exp((lnT - c[0])/c[1])
    Q11 = c[2]*lnT**c[5]*fac/(fac + 1.0/fac) \
        + c[6]*torch.exp(-((lnT - c[7])/c[8])**2) \
        + c[3] + c[9]*lnT**c[4]
    Q11 *= torch.pi
    # Conversion: A^2 -> m^2
    return 1e-20 * Q11

  def _compute_ei_rate(self, T, Te, ve):
    """Electron-ion collision rate (EI)"""
    # Electron and ion number densities
    ne = self.mix.species["em"].n.reshape(1)
    ni = torch.sum(self.mix.species["Arp"].n).reshape(1)
    if self.use_fit:
      # Curve fit model
      return 8.0/3.0 * ve * self._compute_ei_Q11_magin(ne, ni, T, Te)
    else:
      # Analytical table
      return 2.0 * ve * self._compute_ei_Q11_kapper(ne, Te)

  def _compute_ei_Q11_magin(self, ne, ni, T, Te):
    """
    See: Magin's PhD thesis, ULB, 2004
    """
    # Average closest impact parameters for 'em-Arp' and 'em-em' interactions
    f0 = const.UE*const.UE/(8.0*torch.pi*const.UEPS0*const.UKB)
    bh = f0/T
    be = f0/Te
    # Debye shielding distance (em and Arp contributions)
    Ds = self._compute_Ds(ne, ni, T, Te)
    Ds = torch.minimum(Ds,10000.0*(be + bh))
    # Non-dimensional temperature for charge-charge interactions
    Tse = torch.maximum(Ds/(2.0*be), torch.tensor(0.1))
    # Common factors
    lnT1 = torch.log(Tse)
    lnT2 = lnT1*lnT1
    lnT3 = lnT2*lnT1
    lnT4 = lnT3*lnT1
    efac = torch.pi*Ds*Ds/(Tse*Tse)
    # Collision integral for 'em-Arp' and 'em-em' interactions
    c = self.reactions["EI"]["Q11_fit"]
    Q11 = torch.exp(c[0]*lnT4 + c[1]*lnT3 + c[2]*lnT2 + c[3]*lnT1 + c[4])
    return efac * Q11

  def _compute_ei_Q11_kapper(self, ne, Te):
    """
    See: Kapper's PhD thesis, The Ohio State University, 2009
    """
    # Common factors
    T2 = Te*Te
    T3 = T2*Te
    # Compute Coulomb logarithm
    lam = 1.24e7*torch.sqrt(T3/ne)
    # Compute momentum-averaged cross section
    return 5.85e-10*torch.log(lam)/T2

  def _compute_Ds(self, ne, ni, T, Te):
    """Debye shielding distance"""
    f = (const.UEPS0*const.UKB)/(const.UE*const.UE)
    return torch.sqrt(f/(ne/Te + ni/T))

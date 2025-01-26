import torch

from .basic import Basic


class BoxIso(Basic):

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    kin_dtb,
    rad_dtb=None,
    use_rad=False,
    use_proj=False,
    use_factorial=False,
    use_coll_int_fit=False,
    fixed_ne=False
  ):
    super(BoxIso, self).__init__(
      species=species,
      kin_dtb=kin_dtb,
      rad_dtb=rad_dtb,
      use_rad=use_rad,
      use_proj=use_proj,
      use_factorial=use_factorial,
      use_coll_int_fit=use_coll_int_fit
    )
    self.fixed_ne = fixed_ne
    self.nb_eqs = self.nb_comp

  # Function/Jacobian
  # ===================================
  def _fun(self, t, y):
    # ROM activated
    w = self._decode(y) if self.use_rom else y
    # Get number densities
    n = self.mix.get_n(w)
    # Compute sources
    # > Conservative variables
    f_rho = self.sources.call_iso(n)
    # > Primitive variables
    f_w = self.mix.ov_rho * f_rho
    if self.fixed_ne:
      f_w[-1] = 0.0
    # ROM activated
    f = self._encode(f_w) if self.use_rom else f_w
    return f

  # Solving
  # ===================================
  def _set_up(self, y0):
    # Unpack the state vector
    rho = torch.sum(y0[:-2])
    w, T, pe = y0[:-2]/rho, y0[-2], y0[-1]
    # Set density
    self.mix.set_rho(rho)
    # Compute the electron temperature
    n = self.mix.get_n(w)
    Te = self.mix.get_Te(pe, ne=n[-1])
    # Initialize the sources
    self.sources.init_iso(T, Te)
    # Set the function and Jacobian
    self.set_fun_jac()
    return w

  def _encode(self, y):
    return y @ self.P.T if self.use_proj else y @ self.psi

  def _decode(self, z):
    return z @ self.P.T if self.use_proj else z @ self.phi.T

import torch

from typing import Tuple

from torch.overrides import handle_torch_function, has_torch_function


def rsvd(
  X: torch.Tensor,
  Y: torch.Tensor,
  rank: int = 6,
  niter: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  if not torch.jit.is_scripting():
    tensor_ops = (X, Y)
    if not set(map(type, tensor_ops)).issubset(
      (torch.Tensor, type(None))
    ) and has_torch_function(tensor_ops):
      return handle_torch_function(
        rsvd, tensor_ops, X, Y, rank=rank, niter=niter
      )
  return _rsvd(X, Y, rank=rank, niter=niter)

def _rsvd(
  X: torch.Tensor,
  Y: torch.Tensor,
  rank: int = 6,
  niter: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  Randomized SVD for matrix A = Y.T @ X
  Algorithm 5.1 in Halko et al., 2009
  """
  Q = _get_approximate_basis(X, Y, rank, niter=niter)
  B = Q.T @ Y.T @ X
  U, s, Vh = torch.linalg.svd(B, full_matrices=False)
  V = Vh.T
  U = Q.matmul(U)
  return U, s, V

def _get_approximate_basis(
  X: torch.Tensor,
  Y: torch.Tensor,
  rank: int,
  niter: int = 2
) -> torch.Tensor:
  R = torch.randn(X.shape[-1], rank, dtype=X.dtype, device=X.device)
  P = _compute_action(X, Y, R)
  Q = torch.linalg.qr(P).Q
  for _ in range(niter):
    P = _compute_action(X, Y, Q, transpose=True)
    Q = torch.linalg.qr(P).Q
    P = _compute_action(X, Y, Q)
    Q = torch.linalg.qr(P).Q
  return Q

def _compute_action(
  X: torch.Tensor,
  Y: torch.Tensor,
  M: torch.Tensor,
  transpose: bool = False
) -> torch.Tensor:
  if transpose:
    return X.T @ Y @ M
  else:
    return Y.T @ X @ M

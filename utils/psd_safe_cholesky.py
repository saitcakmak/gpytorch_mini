# This is derived from the linear operator implementation.

import warnings
from typing import Optional

import torch
from gpytorch_mini.utils import settings
from gpytorch_mini.utils.exceptions import NanError, NotPSDError
from gpytorch_mini.utils.warnings import NumericalWarning
from torch import Tensor


def psd_safe_cholesky(
    A: Tensor, jitter: Optional[float] = None, max_tries: Optional[int] = None
):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted,
            uses settings.cholesky_jitter.value()
        :attr:`max_tries` (int, optional):
            Number of attempts (with successively increasing jitter) to make before raising an error.
    """
    L, info = torch.linalg.cholesky_ex(A)
    if not torch.any(info):
        return L

    isnan = torch.isnan(A)
    if isnan.any():
        raise NanError(
            f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} "
            f"elements of the {A.shape} tensor are NaN."
        )

    if jitter is None:
        jitter = settings.cholesky_jitter.value(A.dtype)
    if max_tries is None:
        max_tries = settings.cholesky_max_tries.value()
    Aprime = A.clone()
    jitter_prev = 0
    for i in range(max_tries):
        jitter_new = jitter * (10**i)
        # Add jitter only where needed.
        diag_add = (
            ((info > 0) * (jitter_new - jitter_prev))
            .unsqueeze(-1)
            .expand(*Aprime.shape[:-1])
        )
        Aprime.diagonal(dim1=-1, dim2=-2).add_(diag_add)
        jitter_prev = jitter_new
        warnings.warn(
            "Input tensor A is not positive definite. Retrying Cholesky "
            f"decomposition with added jitter of {jitter_new:.1e}.",
            NumericalWarning,
            stacklevel=2,
        )
        L, info = torch.linalg.cholesky_ex(Aprime)
        if not torch.any(info):
            return L
    raise NotPSDError(
        "Matrix is not positive definite after repeatedly adding "
        f"jitter up to {jitter_new:.1e}."
    )

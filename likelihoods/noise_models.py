import warnings
from typing import Any, Optional

import torch
from gpytorch_mini.constraints.constraints import GreaterThan
from gpytorch_mini.distributions.multivariate_normal import MultivariateNormal
from gpytorch_mini.utils import settings
from gpytorch_mini.utils.module import Module
from gpytorch_mini.utils.warnings import NumericalWarning
from torch import Tensor
from torch.nn import Parameter

EMPTY_SIZE = torch.Size([])


class Noise(Module):
    def __call__(
        self, *params: Any, shape: Optional[torch.Size] = None, **kwargs: Any
    ) -> Tensor:
        # For correct typing
        return super().__call__(*params, shape=shape, **kwargs)


class HomoskedasticNoise(Noise):
    def __init__(
        self,
        noise_prior=None,
        noise_constraint=None,
        batch_shape=None,
        num_tasks=1,
    ):
        batch_shape = torch.Size([]) if batch_shape is None else batch_shape
        super().__init__()
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        self.register_parameter(
            name="raw_noise", parameter=Parameter(torch.zeros(*batch_shape, num_tasks))
        )
        if noise_prior is not None:
            self.register_prior(
                "noise_prior", noise_prior, self._noise_param, self._noise_closure
            )

        self.register_constraint("raw_noise", noise_constraint)

    def _noise_param(self, m):
        return m.noise

    def _noise_closure(self, m, v):
        return m._set_noise(v)

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self._set_noise(value)

    def _set_noise(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def forward(
        self, *params: Any, shape: Optional[torch.Size] = None, **kwargs: Any
    ) -> Tensor:
        """In the homoskedastic case, the parameters are only used to infer the required shape.
        Here are the possible scenarios:
        - non-batched noise, non-batched input, non-MT -> noise_diag shape is `n`
        - non-batched noise, non-batched input, MT -> noise_diag shape is `nt`
        - non-batched noise, batched input, non-MT -> noise_diag shape is `b x n` with b' the broadcasted batch shape
        - non-batched noise, batched input, MT -> noise_diag shape is `b x nt`
        - batched noise, non-batched input, non-MT -> noise_diag shape is `b x n`
        - batched noise, non-batched input, MT -> noise_diag shape is `b x nt`
        - batched noise, batched input, non-MT -> noise_diag shape is `b' x n`
        - batched noise, batched input, MT -> noise_diag shape is `b' x nt`
        where `n` is the number of evaluation points and `t` is the number of tasks (i.e. `num_tasks` of self.noise).
        So bascially the shape is always `b' x nt`, with `b'` appropriately broadcast from the noise parameter and
        input batch shapes. `n` and the input batch shape are determined either from the shape arg or from the params
        input. For this it is sufficient to take in a single `shape` arg, with the convention that shape[:-1] is the
        batch shape of the input, and shape[-1] is `n`.

        If a "noise" kwarg (a Tensor) is provided, this noise is used directly.
        """
        # TODO: what are params supposed to be? This seems needlessly too generic.
        if "noise" in kwargs:
            return torch.diag_embed(kwargs["noise"])
        if shape is None:
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = p.shape if len(p.shape) == 1 else p.shape[:-1]
        noise = self.noise
        *batch_shape, n = shape
        noise_batch_shape = noise.shape[:-1] if noise.dim() > 1 else torch.Size()
        num_tasks = noise.shape[-1]
        batch_shape = torch.broadcast_shapes(noise_batch_shape, batch_shape)
        noise = noise.unsqueeze(-2)
        noise_diag = noise.expand(*batch_shape, 1, num_tasks).contiguous()
        if num_tasks == 1:
            # This is batch x 1.
            noise_diag = noise_diag.view(*batch_shape, 1)
        else:
            # This makes it batch x 1 x num_tasks x 1.
            noise_diag = noise_diag.unsqueeze(-1)
        return torch.diag_embed(noise_diag.expand(*noise_diag.shape[:-1], n))


class HeteroskedasticNoise(Noise):
    def __init__(self, noise_model, noise_indices=None, noise_constraint=None):
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)
        super().__init__()
        self.noise_model = noise_model
        self._noise_constraint = noise_constraint
        self._noise_indices = noise_indices

    def forward(
        self,
        *params: Any,
        batch_shape: Optional[torch.Size] = None,
        shape: Optional[torch.Size] = None,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        if noise is not None:
            return torch.diag_embed(noise)
        training = self.noise_model.training  # keep track of mode
        try:
            self.noise_model.eval()  # we want the posterior prediction of the noise model
            with settings.detach_test_caches(False), settings.debug(False):
                if len(params) == 1 and not torch.is_tensor(params[0]):
                    output = self.noise_model(*params[0])
                else:
                    output = self.noise_model(*params)
        finally:
            self.noise_model.train(training)
        if not isinstance(output, MultivariateNormal):
            raise NotImplementedError(
                "Currently only noise models that return a MultivariateNormal are supported"
            )
        # note: this also works with MultitaskMultivariateNormal, where this
        # will return a batched DiagLinearOperators of size n x num_tasks x num_tasks
        noise_diag = (
            output.mean
            if self._noise_indices is None
            else output.mean[..., self._noise_indices]
        )
        return torch.diag_embed(self._noise_constraint.transform(noise_diag))


class FixedGaussianNoise(Noise):
    def __init__(self, noise: Tensor) -> None:
        super().__init__()
        min_noise = settings.min_fixed_noise.value(noise.dtype)
        if noise.lt(min_noise).any():
            warnings.warn(
                "Very small noise values detected. This will likely "
                "lead to numerical instabilities. Rounding small noise "
                f"values up to {min_noise}.",
                NumericalWarning,
                stacklevel=2,
            )
            noise = noise.clamp_min(min_noise)
        self.noise = noise

    def forward(
        self,
        *params: Any,
        shape: Optional[torch.Size] = None,
        noise: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        if shape is None:
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = p.shape if len(p.shape) == 1 else p.shape[:-1]

        if noise is not None:
            return torch.diag_embed(noise)
        elif shape[-1] == self.noise.shape[-1]:
            return torch.diag_embed(self.noise)
        else:
            raise RuntimeError(
                f"The shape of the noise ({self.noise.shape}) does not match the "
                f"shape of the input ({shape})."
            )

    def _apply(self, fn):
        self.noise = fn(self.noise)
        return super(FixedGaussianNoise, self)._apply(fn)

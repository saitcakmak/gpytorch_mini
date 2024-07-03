import math
from copy import deepcopy
from typing import Any, Optional, Union

import torch
from gpytorch_mini.constraints.constraints import Interval
from gpytorch_mini.distributions.multivariate_normal import MultivariateNormal
from gpytorch_mini.likelihoods.likelihood import LikelihoodBase
from gpytorch_mini.likelihoods.noise_models import (
    FixedGaussianNoise,
    HomoskedasticNoise,
    Noise,
)
from gpytorch_mini.priors.prior import Prior
from torch import Tensor
from torch.distributions import Normal


class _GaussianLikelihoodBase(LikelihoodBase):
    """Base class for Gaussian Likelihoods, supporting general heteroskedastic noise models."""

    has_analytic_marginal = True

    def __init__(self, noise_covar: Union[Noise, FixedGaussianNoise]) -> None:
        super().__init__()
        self.noise_covar = noise_covar

    def _shaped_noise_covar(
        self, base_shape: torch.Size, *params: Any, **kwargs: Any
    ) -> Tensor:
        return self.noise_covar(*params, shape=base_shape, **kwargs)

    def expected_log_prob(
        self, target: Tensor, input: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> Tensor:
        noise = self._shaped_noise_covar(input.mean.shape, *params, **kwargs).diagonal(
            dim1=-1, dim2=-2
        )
        # Potentially reshape the noise to deal with the multitask case
        noise = noise.view(*noise.shape[:-1], *input.event_shape)

        # NOTE: this used to have a NaN policy that got removed.
        if torch.isnan(target).any():
            raise RuntimeError("Observations should not contain NaNs. ")
        mean, variance = input.mean, input.variance
        res = (
            ((target - mean).square() + variance) / noise
            + noise.log()
            + math.log(2 * math.pi)
        )
        res = res.mul(-0.5)
        # Do appropriate summation for multitask Gaussian likelihoods
        num_event_dim = len(input.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))

        return res

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> Normal:
        noise = self._shaped_noise_covar(
            function_samples.shape, *params, **kwargs
        ).diagonal(dim1=-1, dim2=-2)
        return Normal(function_samples, noise.sqrt())

    def log_marginal(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *params: Any,
        **kwargs: Any,
    ) -> Tensor:
        marginal = self.marginal(function_dist, *params, **kwargs)
        # NOTE: this used to have a NaN policy that got removed.
        if torch.isnan(observations).any():
            raise RuntimeError("Observations should not contain NaNs. ")
        # We're making everything conditionally independent
        indep_dist = Normal(marginal.mean, marginal.variance.clamp_min(1e-8).sqrt())
        res = indep_dist.log_prob(observations)
        # Do appropriate summation for multitask Gaussian likelihoods
        num_event_dim = len(marginal.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))
        return res

    def marginal(
        self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)


class GaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    The standard likelihood for regression.
    Assumes a standard homoskedastic noise model:

    .. math::
        p(y \mid f) = f + \epsilon, \quad \epsilon \sim \mathcal N (0, \sigma^2)

    where :math:`\sigma^2` is a noise parameter.

    .. note::
        This likelihood can be used for exact or approximate inference.

    .. note::
        GaussianLikelihood has an analytic marginal distribution.

    :param noise_prior: Prior for noise parameter :math:`\sigma^2`.
    :param noise_constraint: Constraint for noise parameter :math:`\sigma^2`.
    :param batch_shape: The batch shape of the learned noise parameter (default: []).
    :param kwargs:

    :ivar torch.Tensor noise: :math:`\sigma^2` parameter (noise)
    """

    def __init__(
        self,
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[Interval] = None,
        batch_shape: Optional[torch.Size] = None,
        **kwargs: Any,
    ) -> None:
        noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            batch_shape=batch_shape,
        )
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)

    def marginal(
        self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> MultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)


class FixedNoiseGaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    A Likelihood that assumes fixed heteroscedastic noise. This is useful when you have fixed, known observation
    noise for each training example.

    Note that this likelihood takes an additional argument when you call it, `noise`, that adds a specified amount
    of noise to the passed MultivariateNormal. This allows for adding known observational noise to test data.

    .. note::
        This likelihood can be used for exact or approximate inference.

    :param noise: Known observation noise (variance) for each training example.
    :type noise: torch.Tensor (... x N)
    :param learn_additional_noise: Set to true if you additionally want to
        learn added diagonal noise, similar to GaussianLikelihood.
    :type learn_additional_noise: bool, optional
    :param batch_shape: The batch shape of the learned noise parameter (default
        []) if :obj:`learn_additional_noise=True`.
    :type batch_shape: torch.Size, optional

    :var torch.Tensor noise: :math:`\sigma^2` parameter (noise)

    .. note::
        FixedNoiseGaussianLikelihood has an analytic marginal distribution.

    Example:
        >>> train_x = torch.randn(55, 2)
        >>> noises = torch.ones(55) * 0.01
        >>> likelihood = FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True)
        >>> pred_y = likelihood(gp_model(train_x))
        >>>
        >>> test_x = torch.randn(21, 2)
        >>> test_noises = torch.ones(21) * 0.02
        >>> pred_y = likelihood(gp_model(test_x), noise=test_noises)
    """

    def __init__(
        self,
        noise: Tensor,
        learn_additional_noise: Optional[bool] = False,
        batch_shape: Optional[torch.Size] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(noise_covar=FixedGaussianNoise(noise=noise))

        self.second_noise_covar: Optional[HomoskedasticNoise] = None
        if learn_additional_noise:
            noise_prior = kwargs.get("noise_prior", None)
            noise_constraint = kwargs.get("noise_constraint", None)
            self.second_noise_covar = HomoskedasticNoise(
                noise_prior=noise_prior,
                noise_constraint=noise_constraint,
                batch_shape=batch_shape,
            )

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise + self.second_noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def second_noise(self) -> Union[float, Tensor]:
        if self.second_noise_covar is None:
            return 0.0
        else:
            return self.second_noise_covar.noise

    @second_noise.setter
    def second_noise(self, value: Tensor) -> None:
        if self.second_noise_covar is None:
            raise RuntimeError(
                "Attempting to set secondary learned noise for FixedNoiseGaussianLikelihood, "
                "but learn_additional_noise must have been False!"
            )
        self.second_noise_covar.initialize(noise=value)

    def get_fantasy_likelihood(self, **kwargs: Any) -> "FixedNoiseGaussianLikelihood":
        if "noise" not in kwargs:
            raise RuntimeError(
                "FixedNoiseGaussianLikelihood.fantasize requires a `noise` kwarg"
            )
        old_noise_covar = self.noise_covar
        self.noise_covar = None  # pyre-fixme[8]
        fantasy_liklihood = deepcopy(self)
        self.noise_covar = old_noise_covar

        old_noise = old_noise_covar.noise
        new_noise = kwargs.get("noise")
        if old_noise.dim() != new_noise.dim():
            old_noise = old_noise.expand(*new_noise.shape[:-1], old_noise.shape[-1])
        fantasy_liklihood.noise_covar = FixedGaussianNoise(
            noise=torch.cat([old_noise, new_noise], -1)
        )
        return fantasy_liklihood

    def _shaped_noise_covar(
        self, base_shape: torch.Size, *params: Any, **kwargs: Any
    ) -> Tensor:
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape

        res = self.noise_covar(*params, shape=shape, **kwargs)

        if self.second_noise_covar is not None:
            res = res + self.second_noise_covar(*params, shape=shape, **kwargs)

        return res

    def marginal(
        self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> MultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        return super().marginal(function_dist, *args, **kwargs)

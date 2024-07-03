import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional, Union

import torch
from gpytorch_mini.distributions.multivariate_normal import MultivariateNormal
from gpytorch_mini.utils import settings
from gpytorch_mini.utils.module import Module
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal


class LikelihoodBase(Module, ABC):
    has_analytic_marginal: bool = False

    def __init__(self, max_plate_nesting: int = 1) -> None:
        super().__init__()
        self.max_plate_nesting: int = max_plate_nesting

    def _draw_likelihood_samples(
        self,
        function_dist: MultivariateNormal,
        *args: Any,
        sample_shape: Optional[torch.Size] = None,
        **kwargs: Any
    ) -> Distribution:
        if sample_shape is None:
            sample_shape = torch.Size(
                [settings.num_likelihood_samples.value()]
                + [1] * (self.max_plate_nesting - len(function_dist.batch_shape) - 1)
            )
        else:
            sample_shape = sample_shape[: -len(function_dist.batch_shape) - 1]
        if self.training:
            num_event_dims = len(function_dist.event_shape)
            function_dist = Normal(function_dist.mean, function_dist.variance.sqrt())
            function_dist = Independent(function_dist, num_event_dims - 1)
        function_samples = function_dist.rsample(sample_shape)
        return self.forward(function_samples, *args, **kwargs)

    def expected_log_prob(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *args: Any,
        **kwargs: Any
    ) -> Tensor:
        likelihood_samples = self._draw_likelihood_samples(
            function_dist, *args, **kwargs
        )
        res = likelihood_samples.log_prob(observations, *args, **kwargs).mean(dim=0)
        return res

    @abstractmethod
    def forward(
        self, function_samples: Tensor, *args: Any, **kwargs: Any
    ) -> Distribution:
        raise NotImplementedError

    def get_fantasy_likelihood(self, **kwargs: Any) -> "LikelihoodBase":
        return deepcopy(self)

    def log_marginal(
        self,
        observations: Tensor,
        function_dist: MultivariateNormal,
        *args: Any,
        **kwargs: Any
    ) -> Tensor:
        likelihood_samples = self._draw_likelihood_samples(
            function_dist, *args, **kwargs
        )
        log_probs = likelihood_samples.log_prob(observations)
        res = log_probs.sub(math.log(log_probs.size(0))).logsumexp(dim=0)
        return res

    def marginal(
        self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> Distribution:
        res = self._draw_likelihood_samples(function_dist, *args, **kwargs)
        return res

    def __call__(
        self, input: Union[Tensor, MultivariateNormal], *args: Any, **kwargs: Any
    ) -> Distribution:
        # Conditional
        if torch.is_tensor(input):
            return super().__call__(input, *args, **kwargs)  # pyre-ignore[7]
        # Marginal
        elif isinstance(input, MultivariateNormal):
            return self.marginal(input, *args, **kwargs)
        # Error
        else:
            raise RuntimeError(
                "Likelihoods expects a MultivariateNormal input to make marginal predictions, or a "
                "torch.Tensor for conditional predictions. Got a {}".format(
                    input.__class__.__name__
                )
            )

from __future__ import annotations

import warnings
from numbers import Number
from typing import Optional, Tuple

import torch
from gpytorch_mini.utils import settings
from gpytorch_mini.utils.psd_safe_cholesky import psd_safe_cholesky
from gpytorch_mini.utils.warnings import NumericalWarning
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal as TMultivariateNormal
from torch.distributions.utils import _standard_normal

EMPTY_SIZE = torch.Size()


class MultivariateNormal(TMultivariateNormal, Distribution):
    """
    Constructs a multivariate normal random variable, based on mean and covariance.
    Can be multivariate, or a batch of multivariate normals

    Passing a vector mean corresponds to a multivariate normal.
    Passing a matrix mean corresponds to a batch of multivariate normals.

    :param mean: `... x N` mean of mvn distribution.
    :param covariance_matrix: `... x N X N` covariance matrix of mvn distribution.
    :param validate_args: If True, validate `mean` and `covariance_matrix` arguments. (Default: False.)

    :ivar torch.Size base_sample_shape: The shape of a base sample (without
        batching) that is used to generate a single sample.
    :ivar torch.Tensor covariance_matrix: The covariance matrix, represented as a dense :class:`torch.Tensor`
    :ivar torch.Tensor mean: The mean.
    :ivar torch.Tensor stddev: The standard deviation.
    :ivar torch.Tensor variance: The variance.
    """

    def __init__(
        self,
        mean: Tensor,
        covariance_matrix: Tensor,
        validate_args: bool = False,
    ) -> None:
        """Initialize the MultivariateNormal distribution from mean
        and the covariance matrix.

        We skip torch MVN initialization here since it converts the covariance
        matrix into scale_tril in __init__ and uses it internally for all computations.
        In doing so, it directly calls Cholesky on the covariance matrix, without
        any protections against numerically non-psd matrices. Here, we instead do this
        lazily, using `psd_safe_cholesky`, the first time `_unbroadcasted_scale_tril`
        is accessed.
        """
        if validate_args:
            if mean.ndim < 1:
                raise ValueError("`mean` must be at least a 1-dimensional tensor.")
            if covariance_matrix.ndim < 2:
                raise ValueError(
                    "`covariance_matrix` must be at least a 2-dimensional tensor."
                )
            if mean.size(-1) != covariance_matrix.size(-1):
                raise ValueError(
                    "`mean` and `covariance_matrix` must have the same "
                    "number of columns."
                )
        batch_shape = torch.broadcast_shapes(
            covariance_matrix.shape[:-2], mean.shape[:-1]
        )
        event_shape = mean.shape[-1:]
        self.loc = mean.expand(batch_shape + (-1,))
        # NOTE: We intentionally don't expand covar here and expand only when the
        # properties are accessed. Cholesky on expanded tensor pays the full cost.
        self._covariance_matrix = covariance_matrix
        self.__unbroadcasted_scale_tril = None
        super(TMultivariateNormal, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def _extended_shape(self, sample_shape: torch.Size = EMPTY_SIZE) -> torch.Size:
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        :param sample_shape: the size of the sample to be drawn.
        """
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        return sample_shape + self._batch_shape + self.base_sample_shape

    @staticmethod
    def _repr_sizes(mean: Tensor, covariance_matrix: Tensor) -> str:
        return (
            f"MultivariateNormal(loc: {mean.size()}, scale: {covariance_matrix.size()})"
        )

    @property
    def _unbroadcasted_scale_tril(self) -> Tensor:
        if self.__unbroadcasted_scale_tril is None:
            self.__unbroadcasted_scale_tril = psd_safe_cholesky(
                self.covariance_matrix,
            ).expand(self.batch_shape + (-1, -1))
        return self.__unbroadcasted_scale_tril

    @_unbroadcasted_scale_tril.setter
    def _unbroadcasted_scale_tril(self, ust: Tensor):
        self.__unbroadcasted_scale_tril = ust

    def add_jitter(self, noise: float = 1e-4) -> MultivariateNormal:
        r"""
        Adds a small constant diagonal to the MVN covariance matrix for numerical stability.

        :param noise: The size of the constant diagonal.
        """
        return self.__class__(
            mean=self.mean,
            covariance_matrix=self.covariance_matrix
            + torch.eye(n=self.covariance_matrix.shape[-1]) * noise,
        )

    @property
    def base_sample_shape(self) -> torch.Size:
        base_sample_shape = self.event_shape

        return base_sample_shape

    @property
    def covariance_matrix(self) -> Tensor:
        # NOTE: This differs from torch MVN, which computes the
        # covariance matrix from the scale_tril. Since we always
        # construct with covariance matrix, we can just return it.
        return self._covariance_matrix.expand(self.batch_shape + (-1, -1))

    def confidence_region(self) -> Tuple[Tensor, Tensor]:
        """
        Returns 2 standard deviations above and below the mean.

        :return: Pair of tensors of size `... x N`, where N is the
            dimensionality of the random variable. The first (second) Tensor is the
            lower (upper) end of the confidence region.
        """
        std2 = self.stddev.mul_(2)
        mean = self.mean
        return mean.sub(std2), mean.add(std2)

    def get_base_samples(self, sample_shape: torch.Size = EMPTY_SIZE) -> Tensor:
        r"""
        Returns i.i.d. standard Normal samples to be used with
        :py:meth:`MultivariateNormal.rsample(base_samples=base_samples)
        <gpytorch_mini.distributions.MultivariateNormal.rsample>`.

        :param sample_shape: The number of samples to generate. (Default: `torch.Size([])`.)
        :return: A `*sample_shape x *batch_shape x N` tensor of i.i.d. standard Normal samples.
        """
        with torch.no_grad():
            shape = self._extended_shape(sample_shape)
            base_samples = _standard_normal(
                shape, dtype=self.loc.dtype, device=self.loc.device
            )
        return base_samples

    def rsample(
        self,
        sample_shape: torch.Size = EMPTY_SIZE,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Generates a `sample_shape` shaped reparameterized sample or `sample_shape`
        shaped batch of reparameterized samples if the distribution parameters
        are batched.

        For the MultivariateNormal distribution, this is accomplished through:

        .. math::
            \boldsymbol \mu + \mathbf L \boldsymbol \epsilon

        where :math:`\boldsymbol \mu \in \mathcal R^N` is the MVN mean,
        :math:`\mathbf L \in \mathcal R^{N \times N}` is a "root" of the
        covariance matrix :math:`\mathbf K` (i.e. :math:`\mathbf L \mathbf
        L^\top = \mathbf K`), and :math:`\boldsymbol \epsilon \in \mathcal R^N` is a
        vector of (approximately) i.i.d. standard Normal random variables.

        :param sample_shape: The number of samples to generate. (Default: `torch.Size([])`.)
        :param base_samples: The `*sample_shape x *batch_shape x N` tensor of
            i.i.d. (or approximately i.i.d.) standard Normal samples to
            reparameterize. (Default: None.)
        :return: A `*sample_shape x *batch_shape x N` tensor of i.i.d. reparameterized samples.
        """
        if base_samples is None:
            base_samples = self.get_base_samples(sample_shape=sample_shape)
        elif self.loc.shape != base_samples.shape[-self.loc.dim() :]:
            # Make sure that the base samples agree with the distribution
            raise RuntimeError(
                "The size of base_samples (minus sample shape dimensions) should "
                f"agree with the size of the distribution. Expected {self.loc.shape} "
                f"but got {base_samples.shape}."
            )

        covar_root = self._unbroadcasted_scale_tril

        # Determine what the appropriate sample_shape parameter is
        sample_shape = base_samples.shape[: base_samples.dim() - self.loc.dim()]

        # Reshape samples to be batch_size x num_dim x num_samples
        # or num_bim x num_samples
        base_samples = base_samples.view(-1, *self.loc.shape[:-1], covar_root.shape[-1])
        base_samples = base_samples.permute(*range(1, self.loc.dim() + 1), 0)

        # Now reparameterize those base samples
        res = covar_root.matmul(base_samples) + self.loc.unsqueeze(-1)

        # Permute and reshape new samples to be original size
        res = res.permute(-1, *range(self.loc.dim())).contiguous()
        res = res.view(sample_shape + self.loc.shape)

        return res

    def sample(
        self,
        sample_shape: torch.Size = EMPTY_SIZE,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Generates a `sample_shape` shaped sample or `sample_shape`
        shaped batch of samples if the distribution parameters
        are batched.

        Note that these samples are not reparameterized and therefore cannot be backpropagated through.

        :param sample_shape: The number of samples to generate. (Default: `torch.Size([])`.)
        :param base_samples: The `*sample_shape x *batch_shape x N` tensor of
            i.i.d. (or approximately i.i.d.) standard Normal samples to
            reparameterize. (Default: None.)
        :return: A `*sample_shape x *batch_shape x N` tensor of i.i.d. samples.
        """
        with torch.no_grad():
            return self.rsample(sample_shape=sample_shape, base_samples=base_samples)

    @property
    def stddev(self) -> Tensor:
        # self.variance is guaranteed to be positive, because we do clamping.
        return self.variance.sqrt()

    @property
    def variance(self) -> Tensor:
        variance = super().variance

        # Check to make sure that variance isn't lower than minimum allowed value (default 1e-6).
        # This ensures that all variances are positive.
        min_variance = settings.min_variance.value(variance.dtype)
        if variance.lt(min_variance).any():
            warnings.warn(
                f"Variance values smaller than {min_variance} are detected. "
                "This is likely due to and may cause numerical instabilities. "
                f"Clamping these variances up to {min_variance}.",
                NumericalWarning,
                stacklevel=2,
            )
            variance = variance.clamp_min(min_variance)
        return variance

    def __add__(self, other: MultivariateNormal) -> MultivariateNormal:
        if isinstance(other, MultivariateNormal):
            return self.__class__(
                mean=self.mean + other.mean,
                covariance_matrix=(self.covariance_matrix + other.covariance_matrix),
            )
        elif isinstance(other, int) or isinstance(other, float):
            return self.__class__(self.mean + other, self.covariance_matrix)
        else:
            raise RuntimeError(
                "Unsupported type {} for addition w/ MultivariateNormal".format(
                    type(other)
                )
            )

    def __mul__(self, other: Number) -> MultivariateNormal:
        if not (isinstance(other, int) or isinstance(other, float)):
            raise RuntimeError("Can only multiply by scalars")
        if other == 1:
            return self
        return self.__class__(
            mean=self.mean * other,
            covariance_matrix=self.covariance_matrix * (other**2),
        )

    def __radd__(self, other: MultivariateNormal) -> MultivariateNormal:
        if other == 0:
            return self
        return self.__add__(other)

    def __truediv__(self, other: Number) -> MultivariateNormal:
        return self.__mul__(1.0 / other)

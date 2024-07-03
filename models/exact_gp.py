import warnings
from copy import deepcopy
from typing import Optional

import torch
from gpytorch_mini.distributions.multivariate_normal import MultivariateNormal
from gpytorch_mini.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch_mini.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch_mini.utils import settings
from gpytorch_mini.utils.module import Module
from gpytorch_mini.utils.warnings import GPInputWarning
from torch import Tensor


class ExactGP(Module):
    r"""
    The base class for any Gaussian process latent function to be used in conjunction
    with exact inference.

    :param torch.Tensor train_inputs: (size n x d) The training features :math:`\mathbf X`.
    :param torch.Tensor train_targets: (size n x m) The training targets :math:`\mathbf y`.
    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood that defines
        the observational distribution. Since we're using exact inference, the likelihood must be Gaussian.

    The :meth:`forward` function should describe how to compute the prior latent distribution
    on a given input. Typically, this will involve a mean and kernel function.
    The result must be a :obj:`~gpytorch.distributions.MultivariateNormal`.

    Calling this model will return the posterior of the latent Gaussian process when conditioned
    on the training data. The output will be a :obj:`~gpytorch.distributions.MultivariateNormal`.

    Example:
        >>> class MyGP(gpytorch.models.ExactGP):
        >>>     def __init__(self, train_x, train_y, likelihood):
        >>>         super().__init__(train_x, train_y, likelihood)
        >>>         self.mean_module = gpytorch.means.ZeroMean()
        >>>         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>>
        >>>     def forward(self, x):
        >>>         mean = self.mean_module(x)
        >>>         covar = self.covar_module(x)
        >>>         return gpytorch.distributions.MultivariateNormal(mean, covar)
        >>>
        >>> # train_x = ...; train_y = ...
        >>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
        >>> model = MyGP(train_x, train_y, likelihood)
        >>>
        >>> # test_x = ...;
        >>> model(test_x)  # Returns the GP latent function at test_x
        >>> likelihood(model(test_x))  # Returns the (approximate) predictive posterior distribution at test_x
    """

    def __init__(
        self,
        train_inputs: Tensor,
        train_targets: Tensor,
        likelihood: _GaussianLikelihoodBase,
    ) -> None:
        # TODO: it'd be great to enforce an explicit output dimension here.
        # Leaving it for later.
        shape = train_inputs.shape[:-1]
        if (
            not isinstance(train_inputs, Tensor)
            or not isinstance(train_targets, Tensor)
            or train_inputs.dim() < 2
            or shape != train_targets.shape[: len(shape)]
        ):
            # TODO: how does BoTorch pass multi-output models in?
            raise RuntimeError(
                "Train inputs and targets must be tensors of shape "
                "(batch x n x d) & (batch x n [x m]), respectively."
            )
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("ExactGP can only handle Gaussian likelihoods")

        super().__init__()
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.likelihood = likelihood
        self.prediction_strategy = None

    def _apply(self, fn):
        # TODO: when is this used?
        self.train_inputs = fn(self.train_inputs)
        self.train_targets = fn(self.train_targets)
        return super()._apply(fn)

    def _clear_cache(self) -> None:
        # The precomputed caches from test time live in prediction_strategy
        self.prediction_strategy = None

    def local_load_samples(self, samples_dict, memo, prefix):
        """
        Replace the model's learned hyperparameters with samples from a posterior distribution.
        """
        # Pyro always puts the samples in the first batch dimension
        num_samples = next(iter(samples_dict.values())).size(0)
        self.train_inputs = self.train_inputs.unsqueeze(0).expand(
            num_samples, *self.train_inputs.shape
        )
        self.train_targets = self.train_targets.unsqueeze(0).expand(
            num_samples, *self.train_targets.shape
        )
        super().local_load_samples(samples_dict, memo, prefix)

    def set_train_data(
        self,
        inputs: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
        strict: bool = True,
    ) -> None:
        """
        Set training data (does not re-fit model hyper-parameters).

        :param torch.Tensor inputs: The new training inputs.
        :param torch.Tensor targets: The new training targets.
        :param bool strict: (default True) If `True`, the new inputs and
            targets must have the same shape, dtype, and device
            as the current inputs and targets. Otherwise, any shape/dtype/device are allowed.
        """
        if inputs is not None:
            if strict:
                for attr in {"shape", "dtype", "device"}:
                    expected_attr = getattr(self.train_inputs, attr, None)
                    found_attr = getattr(inputs, attr, None)
                    if expected_attr != found_attr:
                        msg = "Cannot modify {attr} of inputs (expected {e_attr}, found {f_attr})."
                        msg = msg.format(
                            attr=attr, e_attr=expected_attr, f_attr=found_attr
                        )
                        raise RuntimeError(msg)
            self.train_inputs = inputs
        if targets is not None:
            if strict:
                for attr in {"shape", "dtype", "device"}:
                    expected_attr = getattr(self.train_targets, attr, None)
                    found_attr = getattr(targets, attr, None)
                    if expected_attr != found_attr:
                        msg = "Cannot modify {attr} of targets (expected {e_attr}, found {f_attr})."
                        msg = msg.format(
                            attr=attr, e_attr=expected_attr, f_attr=found_attr
                        )
                        raise RuntimeError(msg)
            self.train_targets = targets
        self.prediction_strategy = None

    def get_fantasy_model(
        self, inputs: Tensor, targets: Tensor, **kwargs
    ):  # TODO: What kwargs?
        """
        Returns a new GP model that incorporates the specified inputs and targets as new training data.

        Using this method is more efficient than updating with `set_train_data` when the number of inputs is relatively
        small, because any computed test-time caches will be updated in linear time rather than computed from scratch.

        .. note::
            If `targets` is a batch (e.g. `b x m`), then the GP returned from this method will be a batch mode GP.
            If `inputs` is of the same (or lesser) dimension as `targets`, then it is assumed that the fantasy points
            are the same for each target batch.

        :param torch.Tensor inputs: (`b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`) Locations of fantasy
            observations.
        :param torch.Tensor targets: (`b1 x ... x bk x m` or `f x b1 x ... x bk x m`) Labels of fantasy observations.
        :return: An `ExactGP` model with `n + m` training examples, where the `m` fantasy examples have been added
            and all test-time caches have been updated.
        :rtype: ~gpytorch.models.ExactGP
        """
        model_batch_shape = self.train_inputs.shape[:-2]
        # TODO: MTMVN
        # if isinstance(self.prediction_strategy.train_prior_dist, MultitaskMultivariateNormal):
        #     data_dim_start = -2
        # else:
        data_dim_start = -1
        target_batch_shape = targets.shape[:data_dim_start]
        input_batch_shape = inputs.shape[:-2]
        tbdim, ibdim = len(target_batch_shape), len(input_batch_shape)

        if not (tbdim == ibdim + 1 or tbdim == ibdim):
            raise RuntimeError(
                f"Unsupported batch shapes: The target batch shape ({target_batch_shape}) must have either the "
                f"same dimension as or one more dimension than the input batch shape ({input_batch_shape})"
            )

        # Check whether we can properly broadcast batch dimensions
        try:
            torch.broadcast_shapes(model_batch_shape, target_batch_shape)
        except RuntimeError:
            raise RuntimeError(
                f"Model batch shape ({model_batch_shape}) and target batch shape "
                f"({target_batch_shape}) are not broadcastable."
            )

        if len(model_batch_shape) > len(input_batch_shape):
            input_batch_shape = model_batch_shape
        if len(model_batch_shape) > len(target_batch_shape):
            target_batch_shape = model_batch_shape

        # If input has no fantasy batch dimension but target does, we can save memory and computation by not
        # computing the covariance for each element of the batch. Therefore we don't expand the inputs to the
        # size of the fantasy model here - this is done below, after the evaluation and fast fantasy update
        train_inputs = self.train_inputs.expand(
            input_batch_shape + self.train_inputs.shape[-2:]
        )
        train_targets = self.train_targets.expand(
            target_batch_shape + self.train_targets.shape[data_dim_start:]
        )

        full_inputs = torch.cat(
            [train_inputs, inputs.expand(input_batch_shape + inputs.shape[-2:])],
            dim=-2,
        )
        full_targets = torch.cat(
            [
                train_targets,
                targets.expand(target_batch_shape + targets.shape[data_dim_start:]),
            ],
            dim=data_dim_start,
        )

        try:
            fantasy_kwargs = {"noise": kwargs.pop("noise")}
        except KeyError:
            fantasy_kwargs = {}

        full_output = super().__call__(full_inputs, **kwargs)

        # Copy model without copying training data or prediction strategy (since we'll overwrite those)
        # TODO: why not just construct a new model? I guess we don't know what the class is.
        old_pred_strat = self.prediction_strategy
        old_train_inputs = self.train_inputs
        old_train_targets = self.train_targets
        old_likelihood = self.likelihood
        self.prediction_strategy = None
        self.train_inputs = None
        self.train_targets = None
        self.likelihood = None
        new_model = deepcopy(self)
        self.prediction_strategy = old_pred_strat
        self.train_inputs = old_train_inputs
        self.train_targets = old_train_targets
        self.likelihood = old_likelihood

        new_model.likelihood = old_likelihood.get_fantasy_likelihood(**fantasy_kwargs)
        if old_pred_strat is not None:
            new_model.prediction_strategy = old_pred_strat.get_fantasy_strategy(
                inputs,
                targets,
                full_inputs,
                full_targets,
                full_output,
                **fantasy_kwargs,
            )

        # if the fantasies are at the same points, we need to expand the inputs for the new model
        if tbdim == ibdim + 1:
            new_model.train_inputs = full_inputs.expand(
                target_batch_shape + full_inputs.shape[-2:]
            )
        else:
            new_model.train_inputs = full_inputs
        new_model.train_targets = full_targets

        return new_model

    def __call__(
        self, inputs: Tensor, **kwargs
    ) -> MultivariateNormal:  # TODO: what kwargs?
        # TODO: Why does this use super.__call__ instead of forward?
        train_inputs = self.train_inputs

        # Training mode: optimizing
        if self.training:
            if settings.debug.on() and not torch.equal(train_inputs, inputs):
                raise RuntimeError("You must train on the training inputs!")
            return super().__call__(inputs, **kwargs)
        # Posterior mode
        else:
            if settings.debug.on() and torch.equal(train_inputs, inputs):
                warnings.warn(
                    "The input matches the stored training data. Did you forget to call model.train()?",
                    GPInputWarning,
                    stacklevel=2,
                )

            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = super().__call__(train_inputs, **kwargs)

                # Create the prediction strategy for
                self.prediction_strategy = DefaultPredictionStrategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            batch_shape = train_inputs.shape[:-2]
            if batch_shape != inputs.shape[:-2]:
                batch_shape = torch.broadcast_shapes(batch_shape, inputs.shape[:-2])
                train_inputs = train_inputs.expand(
                    *batch_shape, *train_inputs.shape[-2:]
                )
                inputs = inputs.expand(*batch_shape, *inputs.shape[-2:])
            full_inputs = torch.cat([train_inputs, inputs], dim=-2)

            # Get the joint distribution for training/test data
            full_output = super().__call__(full_inputs, **kwargs)
            if settings.debug().on() and not isinstance(
                full_output, MultivariateNormal
            ):
                raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            # TODO: Do we want to keep this logic?
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size(
                [joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape]
            )

            # Make the prediction
            (predictive_mean, predictive_covar) = (
                self.prediction_strategy.exact_prediction(full_mean, full_covar)
            )

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(
                *batch_shape, *test_shape
            ).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)

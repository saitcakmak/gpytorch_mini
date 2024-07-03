import string
from typing import Tuple

import torch
from gpytorch_mini.distributions.multivariate_normal import MultivariateNormal
from gpytorch_mini.likelihoods.likelihood import LikelihoodBase
from gpytorch_mini.utils import settings
from gpytorch_mini.utils.psd_safe_cholesky import psd_safe_cholesky
from torch import Tensor


class DefaultPredictionStrategy(object):
    def __init__(
        self,
        train_inputs: Tensor,
        train_prior_dist: MultivariateNormal,
        train_labels: Tensor,
        likelihood: LikelihoodBase,
    ):
        # Get training shape
        self._train_shape = train_prior_dist.event_shape

        # Flatten the training labels
        # TODO: Can we work with the original shapes?
        # This gets rid off the output dimension. Probably useful for MTGPs.
        try:
            flat_train_labels = train_labels.reshape(
                *train_labels.shape[: -len(self.train_shape)], self._train_shape.numel()
            )
        except RuntimeError:
            raise RuntimeError(
                "Flattening the training labels failed. The most common cause of this error is "
                + "that the shapes of the prior mean and the training labels are mismatched. "
                + "The shape of the train targets is {0}, ".format(train_labels.shape)
                + "while the reported shape of the mean is {0}.".format(
                    train_prior_dist.mean.shape
                )
            )

        self.train_inputs = train_inputs
        self.train_prior_dist = train_prior_dist
        self.flat_train_labels = flat_train_labels
        self.likelihood = likelihood
        self.likelihood_train_mvn = self.likelihood(train_prior_dist, train_inputs)
        self._mean_cache = None

    @property
    def num_train(self) -> int:
        return self._train_shape.numel()

    @property
    def train_shape(self) -> torch.Size:
        return self._train_shape

    @property
    def likelihood_train_train_covar(self) -> Tensor:
        return self.likelihood_train_mvn.covariance_matrix

    def __deepcopy__(self, memo):
        # deepcopying prediction strategies of a model evaluated on inputs that require gradients fails
        # with RuntimeError (Only Tensors created explicitly by the user (graph leaves) support the deepcopy
        # protocol at the moment). Overwriting this method make sure that the prediction strategies of a
        # model are set to None upon deepcopying.
        pass

    @property
    def mean_cache(self) -> Tensor:
        if self._mean_cache is None:
            mvn = self.likelihood_train_mvn
            train_labels_offset = (self.flat_train_labels - mvn.loc).unsqueeze(-1)
            self._mean_cache = torch.cholesky_solve(
                train_labels_offset, mvn._unbroadcasted_scale_tril
            ).squeeze(-1)

        mean_cache = self._mean_cache
        if settings.detach_test_caches.on():
            mean_cache = mean_cache.detach()
        return mean_cache

    def exact_prediction(
        self, joint_mean: Tensor, joint_covar: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Find the components of the distribution that contain test data
        test_mean = joint_mean[..., self.num_train :]
        test_test_covar = joint_covar[..., self.num_train :, self.num_train :]
        test_train_covar = joint_covar[..., self.num_train :, : self.num_train]
        return (
            self.exact_predictive_mean(test_mean, test_train_covar),
            self.exact_predictive_covar(test_test_covar, test_train_covar),
        )

    def exact_predictive_mean(
        self, test_mean: Tensor, test_train_covar: Tensor
    ) -> Tensor:
        """
        Computes the posterior predictive covariance of a GP

        :param Tensor test_mean: The test prior mean
        :param ~linear_operator.operators.LinearOperator test_train_covar:
            Covariance matrix between test and train inputs
        :return: The predictive posterior mean of the test points
        """
        # see https://github.com/cornellius-gp/gpytorch/pull/2317#discussion_r1157994719
        mean_cache = self.mean_cache
        if (
            len(mean_cache.shape) == 4
        ):  # TODO: Check if we still need this. Context in above PR.
            mean_cache = mean_cache.squeeze(1)
        return test_mean + (test_train_covar @ mean_cache.unsqueeze(-1)).squeeze(-1)

    def exact_predictive_covar(
        self, test_test_covar: Tensor, test_train_covar: Tensor
    ) -> Tensor:
        """
        Computes the posterior predictive covariance of a GP

        :param ~linear_operator.operators.LinearOperator test_train_covar:
            Covariance matrix between test and train inputs
        :param ~linear_operator.operators.LinearOperator test_test_covar: Covariance matrix between test inputs
        :return: A LinearOperator representing the predictive posterior covariance of the test points
        """
        train_test_covar = test_train_covar.transpose(-1, -2)
        L = self.likelihood_train_mvn._unbroadcasted_scale_tril
        if settings.detach_test_caches.on():
            L = L.detach()
        covar_correction_rhs = torch.cholesky_solve(train_test_covar, L)
        return test_test_covar - test_train_covar @ covar_correction_rhs

    def get_fantasy_strategy(
        self, inputs, targets, full_inputs, full_targets, full_output, **kwargs
    ):
        """
        Returns a new PredictionStrategy that incorporates the specified inputs and targets as new training data.

        This method is primary responsible for updating the mean and covariance caches. To add fantasy data to a
        GP model, use the :meth:`~gpytorch.models.ExactGP.get_fantasy_model` method.

        Args:
            inputs (Tensor `b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`): Locations of fantasy
                observations.
            targets (Tensor `b1 x ... x bk x m` or `f x b1 x ... x bk x m`): Labels of fantasy observations.
            full_inputs (Tensor `b1 x ... x bk x n+m x d` or `f x b1 x ... x bk x n+m x d`): Training data
                concatenated with fantasy inputs
            full_targets (Tensor `b1 x ... x bk x n+m` or `f x b1 x ... x bk x n+m`): Training labels
                concatenated with fantasy labels.
            full_output (:class:`gpytorch.distributions.MultivariateNormal`): Prior called on full_inputs

        Returns:
            A `DefaultPredictionStrategy` model with `n + m` training examples, where the `m` fantasy examples have
            been added and all test-time caches have been updated.
        """
        full_mean, full_covar = full_output.mean, full_output.covariance_matrix
        batch_shape = full_inputs.shape[:-2]
        num_train = self.num_train

        # TODO: MTMVN
        # if isinstance(full_output, MultitaskMultivariateNormal):
        #     num_tasks = full_output.event_shape[-1]
        #     full_mean = full_mean.view(*batch_shape, -1, num_tasks)
        #     fant_mean = full_mean[..., (num_train // num_tasks) :, :]
        #     full_targets = full_targets.view(*targets.shape[:-2], -1)
        # else:
        full_mean = full_mean.view(*batch_shape, -1)
        fant_mean = full_mean[..., num_train:]

        # Evaluate fant x train and fant x fant covariance matrices, leave train x train unevaluated.
        fant_fant_covar = full_covar[..., num_train:, num_train:]
        mvn = self.train_prior_dist.__class__(fant_mean, fant_fant_covar)
        fant_likelihood = self.likelihood.get_fantasy_likelihood(**kwargs)
        mvn_obs = fant_likelihood(mvn, inputs, **kwargs)

        fant_fant_covar = mvn_obs.covariance_matrix
        fant_train_covar = full_covar[..., num_train:, :num_train]

        self.fantasy_inputs = inputs
        self.fantasy_targets = targets

        r"""
        Compute a new mean cache given the old mean cache.

        We have \alpha = K^{-1}y, and we want to solve [K U; U' S][a; b] = [y; y_f], where U' is fant_train_covar,
        S is fant_fant_covar, and y_f is (targets - fant_mean)

        To do this, we solve the bordered linear system of equations for [a; b]:
            AQ = U  # Q = fant_solve
            [S - U'Q]b = y_f - U'\alpha   ==> b = [S - U'Q]^{-1}(y_f - U'\alpha)
            a = \alpha - Qb
        """
        fant_solve = torch.cholesky_solve(
            fant_train_covar.transpose(-2, -1),
            self.likelihood_train_mvn._unbroadcasted_scale_tril,
        )

        # Solve for "b", the lower portion of the *new* \\alpha corresponding to the fantasy points.
        schur_complement = fant_fant_covar - fant_train_covar.matmul(fant_solve)

        # TODO: is this still the case?
        # we'd like to use a less hacky approach for the following, but einsum can be much faster than
        # than unsqueezing/squeezing here (esp. in backward passes), unfortunately it currenlty has some
        # issues with broadcasting: https://github.com/pytorch/pytorch/issues/15671
        prefix = string.ascii_lowercase[
            : max(fant_train_covar.dim() - self.mean_cache.dim() - 1, 0)
        ]
        ftcm = torch.einsum(
            prefix + "...yz,...z->" + prefix + "...y",
            [fant_train_covar, self.mean_cache],
        )

        small_system_rhs = targets - fant_mean - ftcm
        small_system_rhs = small_system_rhs.unsqueeze(-1)
        # Schur complement of a spd matrix is guaranteed to be positive definite
        schur_cholesky = psd_safe_cholesky(schur_complement)
        fant_cache_lower = torch.cholesky_solve(small_system_rhs, schur_cholesky)

        # Get "a", the new upper portion of the cache corresponding to the old training points.
        fant_cache_upper = self.mean_cache.unsqueeze(-1) - fant_solve.matmul(
            fant_cache_lower
        )

        fant_cache_upper = fant_cache_upper.squeeze(-1)
        fant_cache_lower = fant_cache_lower.squeeze(-1)

        # New mean cache.
        fant_mean_cache = torch.cat((fant_cache_upper, fant_cache_lower), dim=-1)

        # Expand inputs accordingly if necessary (for fantasies at the same points)
        # TODO: check this -- full targets might need shape adjustment
        if full_inputs.dim() <= full_targets.dim():
            fant_batch_shape = full_targets.shape[:1]
            full_inputs = full_inputs.expand(fant_batch_shape + full_inputs.shape)
            full_mean = full_mean.expand(fant_batch_shape + full_mean.shape)

        # TODO: MTMVN
        # if isinstance(full_output, MultitaskMultivariateNormal):
        #     full_mean = full_mean.view(*targets.shape[:-2], -1, num_tasks).contiguous()

        # Create new DefaultPredictionStrategy object
        fant_strat = self.__class__(
            train_inputs=full_inputs,
            train_prior_dist=self.train_prior_dist.__class__(full_mean, full_covar),
            train_labels=full_targets,
            likelihood=fant_likelihood,
        )
        fant_strat._mean_cache = fant_mean_cache
        return fant_strat

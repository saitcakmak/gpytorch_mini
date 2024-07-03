import torch
from gpytorch_mini.constraints.constraints import (
    GreaterThan,
    Interval,
    LessThan,
    Positive,
)
from gpytorch_mini.distributions.multivariate_normal import MultivariateNormal
from gpytorch_mini.kernels.matern_kernel import MaternKernel
from gpytorch_mini.kernels.rbf_kernel import RBFKernel
from gpytorch_mini.kernels.scale_kernel import ScaleKernel
from gpytorch_mini.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch_mini.means.constant_mean import ConstantMean
from gpytorch_mini.models.exact_gp import ExactGP
from gpytorch_mini.priors.torch_priors import GammaPrior
from gpytorch_mini.utils.test.base_test_case import BaseTestCase


class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class ExactGPWithPriors(ExactGPModel):
    def __init__(self, train_x, train_y):
        batch_shape = train_x.shape[:-2]
        noise_prior = GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            batch_shape=batch_shape,
            noise_constraint=GreaterThan(
                1e-4,
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=train_x.shape[-1],
                batch_shape=batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
        )


class SumExactGPModel(ExactGPModel):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        covar_a = ScaleKernel(RBFKernel())
        covar_b = ScaleKernel(MaternKernel(nu=0.5))
        self.covar_module = covar_a + covar_b


def get_exact_gp_with_gaussian_likelihood() -> ExactGPModel:
    return ExactGPModel(
        train_x=torch.rand(3, 2),
        train_y=torch.rand(3, 1),
        likelihood=GaussianLikelihood(),
    )

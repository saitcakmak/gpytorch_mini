import torch
from gpytorch_mini.distributions.multivariate_normal import MultivariateNormal
from gpytorch_mini.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch_mini.likelihoods.noise_models import HeteroskedasticNoise
from gpytorch_mini.models.exact_gp import ExactGP
from gpytorch_mini.utils.test.base_test_case import BaseTestCase


class NumericallyUnstableModelExample(ExactGP):
    def __init__(self):
        super().__init__(torch.rand(3, 2), torch.rand(3, 1), GaussianLikelihood())
        self.fail_arithmetic = False

    def train(self, mode=True):
        if mode:
            self.fail_arithmetic = False  # reset on .train()
        super().train(mode=mode)

    def forward(self, x):
        if self.fail_arithmetic:
            raise ArithmeticError()
        return MultivariateNormal(torch.tensor([-3.0]), torch.tensor([[2.0]]))


class TestNoiseModels(BaseTestCase):
    def test_heteroskedasticnoise_error(self) -> None:
        noise_model = NumericallyUnstableModelExample()
        likelihood = HeteroskedasticNoise(noise_model)
        self.assertEqual(noise_model.training, True)
        self.assertEqual(likelihood.training, True)
        noise_model.fail_arithmetic = True
        test_x = torch.tensor([[3.0, 3.0]])
        with self.assertRaises(ArithmeticError):
            likelihood(test_x)
        self.assertEqual(likelihood.training, True)
        likelihood(test_x)

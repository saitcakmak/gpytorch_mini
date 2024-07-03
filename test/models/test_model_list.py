import torch
from gpytorch_mini.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch_mini.models.model_list import IndependentModelList
from gpytorch_mini.utils.test.base_test_case import BaseTestCase
from gpytorch_mini.utils.test.basic_modules import ExactGPModel


N_PTS = 50


class TestModelListGP(BaseTestCase):
    def create_model(self, fixed_noise=False):
        data = torch.randn(N_PTS, 1)
        labels = torch.randn(N_PTS) + 2
        if fixed_noise:
            noise = 0.1 + 0.2 * torch.rand_like(labels)
            likelihood = FixedNoiseGaussianLikelihood(noise)
        else:
            likelihood = GaussianLikelihood()
        return ExactGPModel(data, labels, likelihood)

    def test_forward_eval(self) -> None:
        models = [self.create_model() for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        with self.assertRaises(ValueError):
            model(torch.rand(3, 1))
        model(torch.rand(3, 1), torch.rand(3, 1))

    def test_forward_eval_fixed_noise(self) -> None:
        models = [self.create_model(fixed_noise=True) for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        model(torch.rand(3, 1), torch.rand(3, 1))

    def test_get_fantasy_model(self) -> None:
        models = [self.create_model() for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        model(torch.rand(3, 1), torch.rand(3, 1))
        fant_x = [torch.randn(2, 1), torch.randn(3, 1)]
        fant_y = [torch.randn(2), torch.randn(3)]
        fmodel = model.get_fantasy_model(fant_x, fant_y)
        fmodel(torch.randn(4, 1), torch.randn(4, 1))

    def test_get_fantasy_model_fixed_noise(self) -> None:
        models = [self.create_model(fixed_noise=True) for _ in range(2)]
        model = IndependentModelList(*models)
        model.eval()
        model(torch.rand(3, 1), torch.rand(3, 1))
        fant_x = [torch.randn(2, 1), torch.randn(3, 1)]
        fant_y = [torch.randn(2), torch.randn(3)]
        fant_noise = [0.1 * torch.ones(2), 0.1 * torch.ones(3)]
        fmodel = model.get_fantasy_model(fant_x, fant_y, noise=fant_noise)
        fmodel(torch.randn(4, 1), torch.randn(4, 1))

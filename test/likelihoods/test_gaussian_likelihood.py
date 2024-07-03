import pickle

import torch
from gpytorch_mini.distributions.multivariate_normal import MultivariateNormal
from gpytorch_mini.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from gpytorch_mini.likelihoods.noise_models import FixedGaussianNoise
from gpytorch_mini.priors.torch_priors import GammaPrior
from gpytorch_mini.utils import settings
from gpytorch_mini.utils.test.base_likelihood_test_case import BaseLikelihoodTestCase
from gpytorch_mini.utils.test.base_test_case import BaseTestCase


class TestGaussianLikelihood(BaseLikelihoodTestCase, BaseTestCase):
    seed = 0

    def create_likelihood(self):
        return GaussianLikelihood()

    def test_pickle_with_prior(self) -> None:
        likelihood = GaussianLikelihood(noise_prior=GammaPrior(1, 1))
        pickle.loads(
            pickle.dumps(likelihood)
        )  # Should be able to pickle and unpickle with a prior


class TestGaussianLikelihoodBatch(TestGaussianLikelihood):
    seed = 0

    def create_likelihood(self):
        return GaussianLikelihood(batch_shape=torch.Size([3]))

    def test_nonbatch(self) -> None:
        pass


class TestGaussianLikelihoodMultiBatch(TestGaussianLikelihood):
    seed = 0

    def create_likelihood(self):
        return GaussianLikelihood(batch_shape=torch.Size([2, 3]))

    def test_nonbatch(self) -> None:
        pass

    def test_batch(self) -> None:
        pass


class TestFixedNoiseGaussianLikelihood(BaseLikelihoodTestCase, BaseTestCase):
    def create_likelihood(self):
        noise = 0.1 + torch.rand(5)
        return FixedNoiseGaussianLikelihood(noise=noise)

    def test_fixed_noise_gaussian_likelihood(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            noise = 0.1 + torch.rand(4, device=device, dtype=dtype)
            lkhd = FixedNoiseGaussianLikelihood(noise=noise)
            # test basics
            self.assertIsInstance(lkhd.noise_covar, FixedGaussianNoise)
            self.assertTrue(torch.equal(noise, lkhd.noise))
            new_noise = 0.1 + torch.rand(4, device=device, dtype=dtype)
            lkhd.noise = new_noise
            self.assertTrue(torch.equal(lkhd.noise, new_noise))
            # test __call__
            mean = torch.zeros(4, device=device, dtype=dtype)
            covar = torch.eye(4, device=device, dtype=dtype)
            mvn = MultivariateNormal(mean, covar)
            out = lkhd(mvn)
            self.assertTrue(torch.allclose(out.variance, 1 + new_noise))
            # things should break if dimensions mismatch
            mean = torch.zeros(5, device=device, dtype=dtype)
            covar = torch.eye(5, device=device, dtype=dtype)
            mvn = MultivariateNormal(mean, covar)
            with self.assertRaisesRegex(RuntimeError, "shape of the noise"):
                lkhd(mvn)
            # test __call__ w/ observation noise
            obs_noise = 0.1 + torch.rand(5, device=device, dtype=dtype)
            out = lkhd(mvn, noise=obs_noise)
            self.assertTrue(torch.allclose(out.variance, 1 + obs_noise))
            # test noise smaller than min_fixed_noise
            expected_min_noise = settings.min_fixed_noise.value(dtype)
            noise[:2] = 0
            lkhd = FixedNoiseGaussianLikelihood(noise=noise)
            expected_noise = noise.clone()
            expected_noise[:2] = expected_min_noise
            self.assertTrue(torch.allclose(lkhd.noise, expected_noise))


class TestFixedNoiseGaussianLikelihoodBatch(BaseLikelihoodTestCase, BaseTestCase):
    def create_likelihood(self):
        noise = 0.1 + torch.rand(3, 5)
        return FixedNoiseGaussianLikelihood(noise=noise)

    def test_nonbatch(self) -> None:
        pass


class TestFixedNoiseGaussianLikelihoodMultiBatch(BaseLikelihoodTestCase, BaseTestCase):
    def create_likelihood(self):
        noise = 0.1 + torch.rand(2, 3, 5)
        return FixedNoiseGaussianLikelihood(noise=noise)

    def test_nonbatch(self) -> None:
        pass

    def test_batch(self) -> None:
        pass

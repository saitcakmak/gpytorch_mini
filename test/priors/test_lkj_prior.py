from math import exp

import torch
from gpytorch_mini.priors.lkj_prior import (
    _is_valid_correlation_matrix,
    _is_valid_correlation_matrix_cholesky_factor,
    LKJCholeskyFactorPrior,
    LKJCovariancePrior,
    LKJPrior,
)

from gpytorch_mini.priors.smoothed_box_prior import SmoothedBoxPrior
from gpytorch_mini.utils.test.base_test_case import BaseTestCase
from torch.distributions import LKJCholesky


class TestLKJPrior(BaseTestCase):
    def test_lkj_prior_to_gpu(self) -> None:
        if torch.cuda.is_available():
            prior = LKJPrior(2, 1.0).cuda()
            self.assertEqual(prior.eta.device.type, "cuda")

    def test_lkj_prior_validate_args(self) -> None:
        LKJPrior(2, 1.0, validate_args=True)
        with self.assertRaises(ValueError):
            LKJPrior(1.5, 1.0, validate_args=True)
        with self.assertRaises(ValueError):
            LKJPrior(2, -1.0, validate_args=True)

    def test_lkj_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = LKJPrior(2, torch.tensor(0.5, device=device))
        dist = LKJCholesky(2, torch.tensor(0.5, device=device))

        S = torch.eye(2, device=device)
        S_chol = torch.linalg.cholesky(S)
        self.assertAlmostEqual(prior.log_prob(S), dist.log_prob(S_chol), places=4)
        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        S_chol = torch.linalg.cholesky(S)
        self.assertAllClose(prior.log_prob(S), dist.log_prob(S_chol))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.eye(3, device=device))

    def test_lkj_prior_log_prob_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_lkj_prior_log_prob(cuda=True)

    def test_lkj_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = LKJPrior(2, torch.tensor([0.5, 1.5], device=device))
        dist = LKJCholesky(2, torch.tensor([0.5, 1.5], device=device))

        S = torch.eye(2, device=device)
        S_chol = torch.linalg.cholesky(S)
        self.assertAllClose(prior.log_prob(S), dist.log_prob(S_chol))
        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        S_chol = torch.linalg.cholesky(S)
        self.assertAllClose(prior.log_prob(S), dist.log_prob(S_chol))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.eye(3, device=device))

    def test_lkj_prior_batch_log_prob_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_lkj_prior_batch_log_prob(cuda=True)

    def test_lkj_prior_sample(self, seed=0):
        torch.random.manual_seed(seed)

        prior = LKJPrior(n=5, eta=0.5)
        random_samples = prior.sample(torch.Size((8,)))
        self.assertTrue(_is_valid_correlation_matrix(random_samples))

        max_non_symm = (random_samples - random_samples.transpose(-1, -2)).abs().max()
        self.assertLess(max_non_symm, 1e-4)

        self.assertEqual(random_samples.shape, torch.Size((8, 5, 5)))


class TestLKJCholeskyFactorPrior(BaseTestCase):
    def test_lkj_cholesky_factor_prior_to_gpu(self) -> None:
        if torch.cuda.is_available():
            prior = LKJCholeskyFactorPrior(2, 1.0).cuda()
            self.assertEqual(prior.eta.device.type, "cuda")
            self.assertEqual(prior.C.device.type, "cuda")

    def test_lkj_cholesky_factor_prior_validate_args(self) -> None:
        LKJCholeskyFactorPrior(2, 1.0, validate_args=True)
        with self.assertRaises(ValueError):
            LKJCholeskyFactorPrior(1.5, 1.0, validate_args=True)
        with self.assertRaises(ValueError):
            LKJCholeskyFactorPrior(2, -1.0, validate_args=True)

    def test_lkj_cholesky_factor_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = LKJCholeskyFactorPrior(2, torch.tensor(0.5, device=device))
        dist = LKJCholesky(2, torch.tensor(0.5, device=device))
        S = torch.eye(2, device=device)
        S_chol = torch.linalg.cholesky(S)
        self.assertAlmostEqual(prior.log_prob(S_chol), dist.log_prob(S_chol), places=4)
        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S_chol.device)])
        S_chol = torch.stack([torch.linalg.cholesky(Si) for Si in S])
        self.assertAllClose(prior.log_prob(S_chol), dist.log_prob(S_chol))

    def test_lkj_cholesky_factor_prior_log_prob_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_lkj_cholesky_factor_prior_log_prob(cuda=True)

    def test_lkj_cholesky_factor_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = LKJCholeskyFactorPrior(2, torch.tensor([0.5, 1.5], device=device))
        dist = LKJCholesky(2, torch.tensor([0.5, 1.5], device=device))

        S = torch.eye(2, device=device)
        S_chol = torch.linalg.cholesky(S)
        self.assertAllClose(prior.log_prob(S_chol), dist.log_prob(S_chol))
        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        S_chol = torch.stack([torch.linalg.cholesky(Si) for Si in S])
        self.assertAllClose(prior.log_prob(S_chol), dist.log_prob(S_chol))

    def test_lkj_cholesky_factor_prior_batch_log_prob_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_lkj_cholesky_factor_prior_batch_log_prob(cuda=True)

    def test_lkj_prior_sample(self) -> None:
        prior = LKJCholeskyFactorPrior(2, 0.5)
        random_samples = prior.sample(torch.Size((6,)))
        self.assertTrue(_is_valid_correlation_matrix_cholesky_factor(random_samples))
        self.assertEqual(random_samples.shape, torch.Size((6, 2, 2)))


class TestLKJCovariancePrior(BaseTestCase):
    def test_lkj_covariance_prior_to_gpu(self) -> None:
        if torch.cuda.is_available():
            sd_prior = SmoothedBoxPrior(exp(-1), exp(1))
            prior = LKJCovariancePrior(2, 1.0, sd_prior).cuda()
            self.assertEqual(prior.correlation_prior.eta.device.type, "cuda")
            self.assertEqual(prior.correlation_prior.C.device.type, "cuda")
            self.assertEqual(prior.sd_prior.a.device.type, "cuda")

    def test_lkj_covariance_prior_validate_args(self) -> None:
        sd_prior = SmoothedBoxPrior(exp(-1), exp(1), validate_args=True)
        LKJCovariancePrior(2, 1.0, sd_prior)
        with self.assertRaises(ValueError):
            LKJCovariancePrior(1.5, 1.0, sd_prior, validate_args=True)
        with self.assertRaises(ValueError):
            LKJCovariancePrior(2, -1.0, sd_prior, validate_args=True)

    def test_lkj_covariance_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        sd_prior = SmoothedBoxPrior(exp(-1), exp(1))
        if cuda:
            sd_prior = sd_prior.cuda()
        prior = LKJCovariancePrior(2, torch.tensor(0.5, device=device), sd_prior)
        S = torch.eye(2, device=device)

        corr_dist = LKJCholesky(2, torch.tensor(0.5, device=device))
        dist_log_prob = (
            corr_dist.log_prob(S)
            + sd_prior.log_prob(S.diagonal(dim1=-1, dim2=-2)).sum()
        )
        self.assertAlmostEqual(prior.log_prob(S), dist_log_prob, places=4)

        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        S_chol = torch.linalg.cholesky(S)
        dist_log_prob = corr_dist.log_prob(S_chol) + sd_prior.log_prob(
            torch.diagonal(S, dim1=-2, dim2=-1)
        )
        self.assertAllClose(prior.log_prob(S), dist_log_prob)

    def test_lkj_covariance_prior_log_prob_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_lkj_covariance_prior_log_prob(cuda=True)

    def test_lkj_covariance_prior_log_prob_hetsd(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        a = torch.tensor([exp(-1), exp(-2)], device=device)
        b = torch.tensor([exp(1), exp(2)], device=device)
        sd_prior = SmoothedBoxPrior(a, b)
        prior = LKJCovariancePrior(2, torch.tensor(0.5, device=device), sd_prior)
        corr_dist = LKJCholesky(2, torch.tensor(0.5, device=device))

        S = torch.eye(2, device=device)
        dist_log_prob = (
            corr_dist.log_prob(S)
            + sd_prior.log_prob(S.diagonal(dim1=-1, dim2=-2)).sum()
        )
        self.assertAlmostEqual(prior.log_prob(S), dist_log_prob, places=4)

        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        S_chol = torch.linalg.cholesky(S)
        dist_log_prob = corr_dist.log_prob(S_chol) + sd_prior.log_prob(
            torch.diagonal(S, dim1=-2, dim2=-1)
        )
        self.assertAllClose(prior.log_prob(S), dist_log_prob)

    def test_lkj_covariance_prior_log_prob_hetsd_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_lkj_covariance_prior_log_prob_hetsd(cuda=True)

    def test_lkj_covariance_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        v = torch.ones(2, 1, device=device)
        sd_prior = SmoothedBoxPrior(exp(-1) * v, exp(1) * v)
        prior = LKJCovariancePrior(2, torch.tensor([0.5, 1.5], device=device), sd_prior)
        corr_dist = LKJCholesky(2, torch.tensor([0.5, 1.5], device=device))

        S = torch.eye(2, device=device)
        dist_log_prob = corr_dist.log_prob(S) + sd_prior.log_prob(
            S.diagonal(dim1=-1, dim2=-2)
        )
        self.assertLessEqual((prior.log_prob(S) - dist_log_prob).abs().sum(), 1e-4)

        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        S_chol = torch.linalg.cholesky(S)
        dist_log_prob = corr_dist.log_prob(S_chol) + sd_prior.log_prob(
            torch.diagonal(S, dim1=-2, dim2=-1)
        )
        self.assertLessEqual((prior.log_prob(S) - dist_log_prob).abs().sum(), 1e-4)

    def test_lkj_covariance_prior_batch_log_prob_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_lkj_covariance_prior_batch_log_prob(cuda=True)

    def test_lkj_prior_sample(self) -> None:
        prior = LKJCovariancePrior(2, 0.5, sd_prior=SmoothedBoxPrior(exp(-1), exp(1)))
        random_samples = prior.sample(torch.Size((6,)))
        # need to check that these are positive semi-sefinite
        min_eval = torch.linalg.eigh(random_samples)[0].min()
        self.assertTrue(min_eval >= 0)
        # and that they are symmetric
        max_non_symm = (random_samples - random_samples.transpose(-1, -2)).abs().max()
        self.assertLess(max_non_symm, 1e-4)

        self.assertEqual(random_samples.shape, torch.Size((6, 2, 2)))

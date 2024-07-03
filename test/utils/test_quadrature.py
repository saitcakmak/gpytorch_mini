import os
import random

import torch
from gpytorch_mini.distributions.multivariate_normal import MultivariateNormal
from gpytorch_mini.utils.quadrature import GaussHermiteQuadrature1D
from gpytorch_mini.utils.test.base_test_case import BaseTestCase


class TestQuadrature(BaseTestCase):
    def setUp(self) -> None:
        if (
            os.getenv("UNLOCK_SEED") is None
            or os.getenv("UNLOCK_SEED").lower() == "false"
        ):
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1)
            random.seed(1)

    def tearDown(self) -> None:
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_gauss_hermite_quadrature_1D_normal_nonbatch(self, cuda=False):
        means = torch.randn(10)
        variances = torch.randn(10).abs()
        quadrature = GaussHermiteQuadrature1D()

        if cuda:
            means = means.cuda()
            variances = variances.cuda()
            quadrature = quadrature.cuda()

        dist = torch.distributions.Normal(means, variances.sqrt())

        # Use quadrature
        results = quadrature(torch.sin, dist)

        # Use Monte-Carlo
        samples = dist.rsample(torch.Size([20000]))
        actual = torch.sin(samples).mean(0)

        self.assertLess(torch.mean(torch.abs(actual - results)), 0.1)

    def test_gauss_hermite_quadrature_1D_normal_nonbatch_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_gauss_hermite_quadrature_1D_normal_nonbatch(cuda=True)

    def test_gauss_hermite_quadrature_1D_normal_batch(self, cuda=False):
        means = torch.randn(3, 10)
        variances = torch.randn(3, 10).abs()
        quadrature = GaussHermiteQuadrature1D()

        if cuda:
            means = means.cuda()
            variances = variances.cuda()
            quadrature = quadrature.cuda()

        dist = torch.distributions.Normal(means, variances.sqrt())

        # Use quadrature
        results = quadrature(torch.sin, dist)

        # Use Monte-Carlo
        samples = dist.rsample(torch.Size([20000]))
        actual = torch.sin(samples).mean(0)

        self.assertLess(torch.mean(torch.abs(actual - results)), 0.1)

    def test_gauss_hermite_quadrature_1D_normal_batch_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_gauss_hermite_quadrature_1D_normal_nonbatch(cuda=True)

    def test_gauss_hermite_quadrature_1D_mvn_nonbatch(self, cuda=False):
        means = torch.randn(10)
        variances = torch.randn(10).abs()

        quadrature = GaussHermiteQuadrature1D()

        if cuda:
            means = means.cuda()
            variances = variances.cuda()
            quadrature = quadrature.cuda()

        dist = MultivariateNormal(means, torch.diag_embed(variances.sqrt()))

        # Use quadrature
        results = quadrature(torch.sin, dist)

        # Use Monte-Carlo
        samples = dist.rsample(torch.Size([20000]))
        actual = torch.sin(samples).mean(0)

        self.assertLess(torch.mean(torch.abs(actual - results)), 0.1)

    def test_gauss_hermite_quadrature_1D_mvn_nonbatch_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_gauss_hermite_quadrature_1D_normal_nonbatch(cuda=True)

    def test_gauss_hermite_quadrature_1D_mvn_batch(self, cuda=False):
        means = torch.randn(3, 10)
        variances = torch.randn(3, 10).abs()
        quadrature = GaussHermiteQuadrature1D()

        if cuda:
            means = means.cuda()
            variances = variances.cuda()
            quadrature = quadrature.cuda()

        dist = MultivariateNormal(means, torch.diag_embed(variances.sqrt()))

        # Use quadrature
        results = quadrature(torch.sin, dist)

        # Use Monte-Carlo
        samples = dist.rsample(torch.Size([20000]))
        actual = torch.sin(samples).mean(0)

        self.assertLess(torch.mean(torch.abs(actual - results)), 0.1)

    def test_gauss_hermite_quadrature_1D_mvn_batch_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_gauss_hermite_quadrature_1D_normal_nonbatch(cuda=True)

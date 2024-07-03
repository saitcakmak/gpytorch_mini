import torch
from gpytorch_mini.distributions.multivariate_normal import MultivariateNormal
from gpytorch_mini.utils.test.base_test_case import BaseTestCase
from torch.distributions import MultivariateNormal as TMultivariateNormal


class TestMultivariateNormal(BaseTestCase):
    seed = 1

    def test_multivariate_normal(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0, 1, 2], device=device, dtype=dtype)
            covmat = torch.diag(
                torch.tensor([1, 0.75, 1.5], device=device, dtype=dtype)
            )
            mvn = MultivariateNormal(
                mean=mean, covariance_matrix=covmat, validate_args=True
            )
            self.assertTrue(torch.is_tensor(mvn.covariance_matrix))
            self.assertAllClose(mvn.variance, torch.diag(covmat))
            self.assertAllClose(mvn.scale_tril, covmat.sqrt())
            mvn_plus1 = mvn + 1
            self.assertAllClose(mvn_plus1.mean, mvn.mean + 1)
            self.assertAllClose(mvn_plus1.covariance_matrix, mvn.covariance_matrix)
            mvn_times2 = mvn * 2
            self.assertAllClose(mvn_times2.mean, mvn.mean * 2)
            self.assertAllClose(mvn_times2.covariance_matrix, mvn.covariance_matrix * 4)
            mvn_divby2 = mvn / 2
            self.assertAllClose(mvn_divby2.mean, mvn.mean / 2)
            self.assertAllClose(mvn_divby2.covariance_matrix, mvn.covariance_matrix / 4)
            self.assertAlmostEqual(mvn.entropy().item(), 4.3157, places=4)
            self.assertAlmostEqual(
                mvn.log_prob(torch.zeros(3, device=device, dtype=dtype)).item(),
                -4.8157,
                places=4,
            )
            logprob = mvn.log_prob(torch.zeros(2, 3, device=device, dtype=dtype))
            logprob_expected = torch.tensor(
                [-4.8157, -4.8157], device=device, dtype=dtype
            )
            self.assertAllClose(logprob, logprob_expected)
            conf_lower, conf_upper = mvn.confidence_region()
            self.assertAllClose(conf_lower, mvn.mean - 2 * mvn.stddev)
            self.assertAllClose(conf_upper, mvn.mean + 2 * mvn.stddev)
            self.assertTrue(mvn.sample().shape == torch.Size([3]))
            self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 3]))
            self.assertTrue(
                mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 3])
            )

    def test_multivariate_normal_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_multivariate_normal(cuda=True)

    def test_multivariate_normal_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0, 1, 2], device=device, dtype=dtype)
            covmat = torch.diag(
                torch.tensor([1, 0.75, 1.5], device=device, dtype=dtype)
            )
            mvn = MultivariateNormal(
                mean=mean.repeat(2, 1),
                covariance_matrix=covmat.repeat(2, 1, 1),
                validate_args=True,
            )
            self.assertTrue(torch.is_tensor(mvn.covariance_matrix))
            self.assertAllClose(
                mvn.variance, covmat.diagonal(dim1=-1, dim2=-2).repeat(2, 1)
            )
            self.assertAllClose(
                mvn.scale_tril,
                torch.diag(covmat.diagonal(dim1=-1, dim2=-2).sqrt()).repeat(2, 1, 1),
            )
            mvn_plus1 = mvn + 1
            self.assertAllClose(mvn_plus1.mean, mvn.mean + 1)
            self.assertAllClose(mvn_plus1.covariance_matrix, mvn.covariance_matrix)
            mvn_times2 = mvn * 2
            self.assertAllClose(mvn_times2.mean, mvn.mean * 2)
            self.assertAllClose(mvn_times2.covariance_matrix, mvn.covariance_matrix * 4)
            mvn_divby2 = mvn / 2
            self.assertAllClose(mvn_divby2.mean, mvn.mean / 2)
            self.assertAllClose(mvn_divby2.covariance_matrix, mvn.covariance_matrix / 4)
            self.assertAllClose(
                mvn.entropy(), 4.3157 * torch.ones(2, device=device, dtype=dtype)
            )
            logprob = mvn.log_prob(torch.zeros(2, 3, device=device, dtype=dtype))
            logprob_expected = -4.8157 * torch.ones(2, device=device, dtype=dtype)
            self.assertAllClose(logprob, logprob_expected)
            logprob = mvn.log_prob(torch.zeros(2, 2, 3, device=device, dtype=dtype))
            logprob_expected = -4.8157 * torch.ones(2, 2, device=device, dtype=dtype)
            self.assertAllClose(logprob, logprob_expected)
            conf_lower, conf_upper = mvn.confidence_region()
            self.assertAllClose(conf_lower, mvn.mean - 2 * mvn.stddev)
            self.assertAllClose(conf_upper, mvn.mean + 2 * mvn.stddev)
            self.assertTrue(mvn.sample().shape == torch.Size([2, 3]))
            self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 2, 3]))
            self.assertTrue(
                mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 2, 3])
            )

    def test_multivariate_normal_batch_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_multivariate_normal_batch(cuda=True)

    def test_multivariate_normal_correlated_samples(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0, 1, 2], device=device, dtype=dtype)
            covmat = torch.diag(
                torch.tensor([1, 0.75, 1.5], device=device, dtype=dtype)
            )
            mvn = MultivariateNormal(mean=mean, covariance_matrix=covmat)
            base_samples = mvn.get_base_samples(torch.Size([3, 4]))
            self.assertTrue(
                mvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 3])
            )
            base_samples = mvn.get_base_samples()
            self.assertTrue(
                mvn.sample(base_samples=base_samples).shape == torch.Size([3])
            )

    def test_multivariate_normal_correlated_samples_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_multivariate_normal_correlated_samples(cuda=True)

    def test_multivariate_normal_batch_correlated_samples(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.tensor([0, 1, 2], device=device, dtype=dtype)
            covmat = torch.diag(
                torch.tensor([1, 0.75, 1.5], device=device, dtype=dtype)
            )
            mvn = MultivariateNormal(
                mean=mean.repeat(2, 1),
                covariance_matrix=covmat.repeat(2, 1, 1),
            )
            base_samples = mvn.get_base_samples(torch.Size((3, 4)))
            self.assertTrue(
                mvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 2, 3])
            )
            base_samples = mvn.get_base_samples()
            self.assertTrue(
                mvn.sample(base_samples=base_samples).shape == torch.Size([2, 3])
            )

    def test_multivariate_normal_batch_correlated_samples_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_multivariate_normal_batch_correlated_samples(cuda=True)

    def test_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        for dtype in (torch.float, torch.double):
            mean = torch.randn(4, device=device, dtype=dtype)
            var = torch.randn(4, device=device, dtype=dtype).abs_()
            values = torch.randn(4, device=device, dtype=dtype)

            res = MultivariateNormal(mean, torch.diag_embed(var)).log_prob(values)
            actual = TMultivariateNormal(
                mean, torch.eye(4, device=device, dtype=dtype) * var
            ).log_prob(values)
            self.assertLess((res - actual).div(res).abs().item(), 1e-2)

            mean = torch.randn(3, 4, device=device, dtype=dtype)
            var = torch.randn(3, 4, device=device, dtype=dtype).abs_()
            values = torch.randn(3, 4, device=device, dtype=dtype)

            res = MultivariateNormal(mean, torch.diag_embed(var)).log_prob(values)
            actual = TMultivariateNormal(
                mean,
                var.unsqueeze(-1)
                * torch.eye(4, device=device, dtype=dtype).repeat(3, 1, 1),
            ).log_prob(values)
            self.assertLess((res - actual).div(res).abs().norm(), 1e-2)

    def test_log_prob_cuda(self) -> None:
        if torch.cuda.is_available():
            self.test_log_prob(cuda=True)

    def test_base_sample_shape(self) -> None:
        a = torch.rand(10, 5)
        dist = MultivariateNormal(torch.zeros(5), torch.diag_embed(a))
        samples = dist.rsample(torch.Size((16,)), base_samples=torch.randn(16, 10, 5))
        self.assertEqual(samples.shape, torch.Size((16, 10, 5)))

        # Wrong shape of base samples
        self.assertRaises(
            RuntimeError,
            dist.rsample,
            torch.Size((16,)),
            base_samples=torch.rand(16, 10, 10),
        )

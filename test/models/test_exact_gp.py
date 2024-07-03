import torch
from gpytorch_mini.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch_mini.utils.test.base_test_case import BaseTestCase
from gpytorch_mini.utils.test.basic_modules import ExactGPModel, SumExactGPModel
from gpytorch_mini.utils.test.model_test_case import BaseModelTestCase

N_PTS = 50


class TestExactGP(BaseModelTestCase, BaseTestCase):
    def create_model(self, train_x, train_y, likelihood):
        model = ExactGPModel(train_x, train_y, likelihood)
        return model

    def create_test_data(self):
        return torch.randn(N_PTS, 1)

    def create_likelihood_and_labels(self):
        likelihood = GaussianLikelihood()
        labels = torch.randn(N_PTS) + 2
        return likelihood, labels

    def create_batch_test_data(self, batch_shape):
        return torch.randn(*batch_shape, N_PTS, 1)

    def create_batch_likelihood_and_labels(self, batch_shape):
        likelihood = GaussianLikelihood(batch_shape=batch_shape)
        labels = torch.randn(*batch_shape, N_PTS) + 2
        return likelihood, labels

    def test_batch_forward_then_nonbatch_forward_eval(self) -> None:
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([3]))
        likelihood, labels = self.create_batch_likelihood_and_labels(
            batch_shape=torch.Size([3])
        )
        model = self.create_model(batch_data, labels, likelihood)
        model.eval()
        output = model(batch_data)

        # Smoke test derivatives working
        output.mean.sum().backward()

        self.assertTrue(output.covariance_matrix.dim() == 3)
        self.assertTrue(output.covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.covariance_matrix.size(-2) == batch_data.size(-2))

        # Create non-batch data
        data = self.create_test_data()
        output = model(data)
        self.assertTrue(output.covariance_matrix.dim() == 3)
        self.assertTrue(output.covariance_matrix.size(-1) == data.size(-2))
        self.assertTrue(output.covariance_matrix.size(-2) == data.size(-2))

        # Smoke test derivatives working
        output.mean.sum().backward()

    def test_batch_forward_then_different_batch_forward_eval(self) -> None:
        non_batch_data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(non_batch_data, labels, likelihood)
        model.eval()

        # Batch size 3
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([3]))
        output = model(batch_data)
        self.assertTrue(output.covariance_matrix.dim() == 3)
        self.assertTrue(output.covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.covariance_matrix.size(-2) == batch_data.size(-2))

        # Now Batch size 2
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([2]))
        output = model(batch_data)
        self.assertTrue(output.covariance_matrix.dim() == 3)
        self.assertTrue(output.covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.covariance_matrix.size(-2) == batch_data.size(-2))

        # Now 3 again
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([3]))
        output = model(batch_data)
        self.assertTrue(output.covariance_matrix.dim() == 3)
        self.assertTrue(output.covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.covariance_matrix.size(-2) == batch_data.size(-2))

        # Now 1
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([1]))
        output = model(batch_data)
        self.assertTrue(output.covariance_matrix.dim() == 3)
        self.assertTrue(output.covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.covariance_matrix.size(-2) == batch_data.size(-2))


class TestSumExactGP(TestExactGP):
    def create_model(self, train_x, train_y, likelihood):
        model = SumExactGPModel(train_x, train_y, likelihood)
        return model

from abc import abstractmethod

import torch


class BaseModelTestCase(object):
    @abstractmethod
    def create_model(self, train_x, train_y, likelihood): ...

    @abstractmethod
    def create_test_data(self) -> None: ...

    @abstractmethod
    def create_likelihood_and_labels(self) -> None: ...

    @abstractmethod
    def create_batch_test_data(self, batch_shape): ...

    @abstractmethod
    def create_batch_likelihood_and_labels(self, batch_shape): ...

    def test_forward_train(self) -> None:
        data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(data, labels, likelihood)
        model.train()
        output = model(data)
        self.assertTrue(output.covariance_matrix.dim() == 2)
        self.assertTrue(output.covariance_matrix.size(-1) == data.size(-2))
        self.assertTrue(output.covariance_matrix.size(-2) == data.size(-2))

    def test_batch_forward_train(self) -> None:
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([3]))
        likelihood, labels = self.create_batch_likelihood_and_labels(
            batch_shape=torch.Size([3])
        )
        model = self.create_model(batch_data, labels, likelihood)
        model.train()
        output = model(batch_data)
        self.assertTrue(output.covariance_matrix.dim() == 3)
        self.assertTrue(output.covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.covariance_matrix.size(-2) == batch_data.size(-2))

    def test_multi_batch_forward_train(self) -> None:
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([2, 3]))
        likelihood, labels = self.create_batch_likelihood_and_labels(
            batch_shape=torch.Size([2, 3])
        )
        model = self.create_model(batch_data, labels, likelihood)
        model.train()
        output = model(batch_data)
        self.assertTrue(output.covariance_matrix.dim() == 4)
        self.assertTrue(output.covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.covariance_matrix.size(-2) == batch_data.size(-2))

    def test_forward_eval(self) -> None:
        data = self.create_test_data()
        likelihood, labels = self.create_likelihood_and_labels()
        model = self.create_model(data, labels, likelihood)
        model.eval()
        output = model(data)
        self.assertTrue(output.covariance_matrix.dim() == 2)
        self.assertTrue(output.covariance_matrix.size(-1) == data.size(-2))
        self.assertTrue(output.covariance_matrix.size(-2) == data.size(-2))

    def test_batch_forward_eval(self) -> None:
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([3]))
        likelihood, labels = self.create_batch_likelihood_and_labels(
            batch_shape=torch.Size([3])
        )
        model = self.create_model(batch_data, labels, likelihood)
        model.eval()
        output = model(batch_data)
        self.assertTrue(output.covariance_matrix.dim() == 3)
        self.assertTrue(output.covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.covariance_matrix.size(-2) == batch_data.size(-2))

    def test_multi_batch_forward_eval(self) -> None:
        batch_data = self.create_batch_test_data(batch_shape=torch.Size([2, 3]))
        likelihood, labels = self.create_batch_likelihood_and_labels(
            batch_shape=torch.Size([2, 3])
        )
        model = self.create_model(batch_data, labels, likelihood)
        model.eval()
        output = model(batch_data)
        self.assertTrue(output.covariance_matrix.dim() == 4)
        self.assertTrue(output.covariance_matrix.size(-1) == batch_data.size(-2))
        self.assertTrue(output.covariance_matrix.size(-2) == batch_data.size(-2))

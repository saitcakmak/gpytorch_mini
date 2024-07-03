import pickle

import torch
from gpytorch_mini.constraints.constraints import GreaterThan
from gpytorch_mini.means.constant_mean import ConstantMean
from gpytorch_mini.priors.torch_priors import NormalPrior
from gpytorch_mini.utils.test.base_mean_test_case import BaseMeanTestCase
from gpytorch_mini.utils.test.base_test_case import BaseTestCase


class TestConstantMean(BaseMeanTestCase, BaseTestCase):
    batch_shape = None

    def create_mean(self, prior=None, constraint=None):
        return ConstantMean(
            constant_prior=prior,
            constant_constraint=constraint,
            batch_shape=(self.__class__.batch_shape or torch.Size([])),
        )

    def test_prior(self) -> None:
        if self.batch_shape is None:
            prior = NormalPrior(0.0, 1.0)
        else:
            prior = NormalPrior(
                torch.zeros(self.batch_shape), torch.ones(self.batch_shape)
            )
        mean = self.create_mean(prior=prior)
        self.assertEqual(mean.mean_prior, prior)
        pickle.loads(
            pickle.dumps(mean)
        )  # Should be able to pickle and unpickle with a prior
        value = prior.sample()
        mean._constant_closure(mean, value)
        self.assertTrue(
            torch.equal(mean.constant.data, value.reshape(mean.constant.data.shape))
        )

    def test_constraint(self) -> None:
        mean = self.create_mean()
        self.assertAllClose(mean.constant, torch.zeros(mean.constant.shape))

        constraint = GreaterThan(1.5)
        mean = self.create_mean(constraint=constraint)
        self.assertTrue(torch.all(mean.constant >= 1.5))
        mean.constant = torch.full(
            self.__class__.batch_shape or torch.Size([]), fill_value=1.65
        )
        self.assertAllClose(
            mean.constant, torch.tensor(1.65).expand(mean.constant.shape)
        )


class TestConstantMeanBatch(TestConstantMean):
    batch_shape = torch.Size([3])


class TestConstantMeanMultiBatch(TestConstantMean):
    batch_shape = torch.Size([2, 3])

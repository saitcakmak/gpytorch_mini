from gpytorch_mini.kernels.index_kernel import IndexKernel
from gpytorch_mini.priors.torch_priors import NormalPrior
from gpytorch_mini.utils.test.base_test_case import BaseTestCase


class TestIndexKernel(BaseTestCase):
    def create_kernel_with_prior(self, prior):
        return IndexKernel(num_tasks=1, prior=prior)

    def test_prior_type(self) -> None:
        """
        Raising TypeError if prior type is other than gpytorch_mini.priors.Prior
        """
        self.create_kernel_with_prior(None)
        self.create_kernel_with_prior(NormalPrior(0, 1))
        self.assertRaises(TypeError, self.create_kernel_with_prior, 1)

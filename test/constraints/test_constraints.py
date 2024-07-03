import math

import torch
from gpytorch_mini.constraints.constraints import (
    GreaterThan,
    Interval,
    LessThan,
    Positive,
)
from gpytorch_mini.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch_mini.utils.test.base_test_case import BaseTestCase
from gpytorch_mini.utils.test.basic_modules import get_exact_gp_with_gaussian_likelihood
from torch import sigmoid
from torch.nn.functional import softplus


class TestInterval(BaseTestCase):
    def test_transform_float_bounds(self) -> None:
        constraint = Interval(1.0, 5.0)

        v = torch.tensor(-3.0)

        value = constraint.transform(v)
        actual_value = ((5.0 - 1.0) * sigmoid(v)) + 1.0

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_bounds(self) -> None:
        constraint = Interval(1.0, 5.0)

        v = torch.tensor(-3.0)

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(v, value)

    def test_transform_tensor_bounds(self) -> None:
        constraint = Interval(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))

        v = torch.tensor([-3.0, -2.0])

        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = (3.0 - 1.0) * sigmoid(v[0]) + 1.0
        actual_value[1] = (4.0 - 2.0) * sigmoid(v[1]) + 2.0

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_tensor_bounds(self) -> None:
        constraint = Interval(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))

        v = torch.tensor([-3.0, -2.0])

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(v, value)

    def test_initial_value(self) -> None:
        constraint = Interval(1.0, 5.0, transform=None, initial_value=3.0)
        lkhd = GaussianLikelihood(noise_constraint=constraint)
        self.assertEqual(lkhd.noise.item(), 3.0)

    def test_error_on_infinite(self) -> None:
        err_msg = "Cannot make an Interval directly with non-finite bounds"
        with self.assertRaisesRegex(ValueError, err_msg):
            Interval(0.0, math.inf)
        with self.assertRaisesRegex(ValueError, err_msg):
            Interval(-math.inf, 0.0)


class TestGreaterThan(BaseTestCase):
    def test_transform_float_greater_than(self) -> None:
        constraint = GreaterThan(1.0)

        v = torch.tensor(-3.0)

        value = constraint.transform(v)
        actual_value = softplus(v) + 1.0

        self.assertAllClose(value, actual_value)

    def test_transform_tensor_greater_than(self) -> None:
        constraint = GreaterThan([1.0, 2.0])

        v = torch.tensor([-3.0, -2.0])

        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = softplus(v[0]) + 1.0
        actual_value[1] = softplus(v[1]) + 2.0

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_greater_than(self) -> None:
        constraint = GreaterThan(1.0)

        v = torch.tensor(-3.0)

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)

    def test_inverse_transform_tensor_greater_than(self) -> None:
        constraint = GreaterThan([1.0, 2.0])

        v = torch.tensor([-3.0, -2.0])

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)


class TestLessThan(BaseTestCase):
    def test_transform_float_less_than(self) -> None:
        constraint = LessThan(1.0)

        v = torch.tensor(-3.0)

        value = constraint.transform(v)
        actual_value = -softplus(-v) + 1.0

        self.assertAllClose(value, actual_value)

    def test_transform_tensor_less_than(self) -> None:
        constraint = LessThan([1.0, 2.0])

        v = torch.tensor([-3.0, -2.0])

        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = -softplus(-v[0]) + 1.0
        actual_value[1] = -softplus(-v[1]) + 2.0

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_less_than(self) -> None:
        constraint = LessThan(1.0)

        v = torch.tensor(-3.0)

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)

    def test_inverse_transform_tensor_less_than(self) -> None:
        constraint = LessThan([1.0, 2.0])

        v = torch.tensor([-3.0, -2.0])

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)


class TestPositive(BaseTestCase):
    def test_transform_float_positive(self) -> None:
        constraint = Positive()

        v = torch.tensor(-3.0)

        value = constraint.transform(v)
        actual_value = softplus(v)

        self.assertAllClose(value, actual_value)

    def test_transform_tensor_positive(self) -> None:
        constraint = Positive()

        v = torch.tensor([-3.0, -2.0])

        value = constraint.transform(v)
        actual_value = v.clone()
        actual_value[0] = softplus(v[0])
        actual_value[1] = softplus(v[1])

        self.assertAllClose(value, actual_value)

    def test_inverse_transform_float_positive(self) -> None:
        constraint = Positive()

        v = torch.tensor(-3.0)

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)

    def test_inverse_transform_tensor_positive(self) -> None:
        constraint = Positive()

        v = torch.tensor([-3.0, -2.0])

        value = constraint.inverse_transform(constraint.transform(v))

        self.assertAllClose(value, v)


class TestConstraintNaming(BaseTestCase):
    def test_constraint_by_name(self) -> None:
        model = get_exact_gp_with_gaussian_likelihood()

        constraint = model.constraint_for_parameter_name(
            "likelihood.noise_covar.raw_noise"
        )
        self.assertIsInstance(constraint, GreaterThan)

        constraint = model.constraint_for_parameter_name(
            "covar_module.base_kernel.raw_lengthscale"
        )
        self.assertIsInstance(constraint, Positive)

        constraint = model.constraint_for_parameter_name("mean_module.constant")
        self.assertIsNone(constraint)

    def test_named_parameters_and_constraints(self) -> None:
        model = get_exact_gp_with_gaussian_likelihood()

        for name, _param, constraint in model.named_parameters_and_constraints():
            if name == "likelihood.noise_covar.raw_noise":
                self.assertIsInstance(constraint, GreaterThan)
            elif name == "mean_module.constant":
                self.assertIsNone(constraint)
            elif name == "covar_module.raw_outputscale":
                self.assertIsInstance(constraint, Positive)
            elif name == "covar_module.base_kernel.raw_lengthscale":
                self.assertIsInstance(constraint, Positive)

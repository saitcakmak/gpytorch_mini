from typing import Any, Optional

import torch

from gpytorch_mini.constraints.constraints import Interval
from gpytorch_mini.means.mean import Mean
from gpytorch_mini.priors.prior import Prior

EMPTY_SIZE = torch.Size()


class ConstantMean(Mean):
    r"""
    A (non-zero) constant prior mean function, i.e.:

    .. math::
        \mu(\mathbf x) = C

    where :math:`C` is a learned constant.

    :param constant_prior: Prior for constant parameter :math:`C`.
    :type constant_prior: ~gpytorch_mini.priors.Prior, optional
    :param constant_constraint: Constraint for constant parameter :math:`C`.
    :type constant_constraint: ~gpytorch_mini.priors.Interval, optional
    :param batch_shape: The batch shape of the learned constant(s) (default: []).
    :type batch_shape: torch.Size, optional

    :var torch.Tensor constant: :math:`C` parameter
    """

    def __init__(
        self,
        constant_prior: Optional[Prior] = None,
        constant_constraint: Optional[Interval] = None,
        batch_shape: torch.Size = EMPTY_SIZE,
        **kwargs: Any,
    ):
        super(ConstantMean, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(
            name="raw_constant", parameter=torch.nn.Parameter(torch.zeros(batch_shape))
        )
        if constant_prior is not None:
            self.register_prior(
                "mean_prior",
                constant_prior,
                self._constant_param,
                self._constant_closure,
            )
        if constant_constraint is not None:
            self.register_constraint("raw_constant", constant_constraint)

    @property
    def constant(self):
        return self._constant_param(self)

    @constant.setter
    def constant(self, value):
        self._constant_closure(self, value)

    # We need a getter of this form so that we can pickle ConstantMean modules with a mean prior, see PR #1992
    def _constant_param(self, m):
        if hasattr(m, "raw_constant_constraint"):
            return m.raw_constant_constraint.transform(m.raw_constant)
        return m.raw_constant

    # We need a setter of this form so that we can pickle ConstantMean modules with a mean prior, see PR #1992
    def _constant_closure(self, m, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(m.raw_constant)

        if hasattr(m, "raw_constant_constraint"):
            m.initialize(
                raw_constant=m.raw_constant_constraint.inverse_transform(value)
            )
        else:
            m.initialize(raw_constant=value)

    def forward(self, input):
        constant = self.constant.unsqueeze(-1)  # *batch_shape x 1
        return constant.expand(torch.broadcast_shapes(constant.shape, input.shape[:-1]))

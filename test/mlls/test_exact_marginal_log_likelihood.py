import torch
from gpytorch_mini.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from gpytorch_mini.utils.test.base_test_case import BaseTestCase
from gpytorch_mini.utils.test.basic_modules import ExactGPWithPriors


class TestExactMarginalLogLikelihood(BaseTestCase):
    def test_batched_eval(self) -> None:
        train_x = torch.rand(10, 2)
        train_y = torch.randn(10)
        non_batch_model = ExactGPWithPriors(train_x, train_y)
        mll = ExactMarginalLogLikelihood(non_batch_model.likelihood, non_batch_model)
        output = non_batch_model(train_x)
        non_batch_mll_eval = mll(output, train_y)

        train_x = train_x.expand(10, -1, -1)
        train_y = train_y.expand(10, -1)
        batch_model = ExactGPWithPriors(train_x, train_y)
        mll = ExactMarginalLogLikelihood(batch_model.likelihood, batch_model)
        output = batch_model(train_x)
        batch_mll_eval = mll(output, train_y)

        self.assertEqual(non_batch_mll_eval.shape, torch.Size())
        self.assertEqual(batch_mll_eval.shape, torch.Size([10]))
        self.assertTrue(torch.allclose(non_batch_mll_eval.expand(10), batch_mll_eval))

    def test_mll_computation(self) -> None:
        train_x, train_y = (torch.rand(10, 2), torch.rand(10))
        model = ExactGPWithPriors(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        output = model(train_x)
        marginal_log_likelihood = mll(output, train_y)

        marginal_likelihood = model.likelihood(output)
        noise_prior = next(model.likelihood.named_priors())[2]
        outputscale_prior = next(model.covar_module.named_priors())[2]
        lengthscale_prior = next(model.covar_module.base_kernel.named_priors())[2]

        log_probs = [
            marginal_likelihood.log_prob(train_y),
            noise_prior.log_prob(model.likelihood.noise),
            outputscale_prior.log_prob(model.covar_module.outputscale),
            lengthscale_prior.log_prob(
                model.covar_module.base_kernel.lengthscale
            ).sum(),
        ]
        marginal_log_likelihood_by_hand = sum(log_probs) / train_y.shape[0]

        self.assertTrue(
            torch.allclose(marginal_log_likelihood, marginal_log_likelihood_by_hand)
        )

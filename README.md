## GPyTorch mini

This is a limited, experimental implementation of GPyTorch that does not utilize the linear operator library.

Key changes:
- Removed any dependency on linear operator.
- Replaced obscure @cached decorator with explicit attributes for caches on the objects.
- Reimplement the linear algebra (for MVN & exact prediction strategy) directly using torch operations rather than relying on the linear operator package.
- The size of GPyTorch mini is significantly smaller (~20% of the original) since it does not include any approximate models or other modules for non-exact GPs.

High level summary of the observations:
- For small models evaluated with small tensors, GPyTorch mini is up to 2x faster.
- As the data size increases, linear algebra becomes the bottleneck and the gap closes. For large tensors, the two models have similar runtimes.
- When using these models with acquisition functions like qNEI (in BoTorch), the runtimes are comparable.
- The runtime improvements for the small models are negligible when viewed in the context of e2e acquisition evaluation.
- When used e2e in Ax benchmarks, we do not observe a consistent change in fit or gen times. There are some slight differences, though we also observe similar differences between two identical runs.
- Neither GPyTorch nor GPyTorch mini models show any benefit from using torch.compile on CPU. I don’t think torch.compile does anything on the CPU with the default settings. This requires further investigation.

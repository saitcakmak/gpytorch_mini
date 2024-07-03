import torch


class _dtype_value_context:
    _global_float_value = None
    _global_double_value = None
    _global_half_value = None

    @classmethod
    def value(cls, dtype):
        if torch.is_tensor(dtype):
            dtype = dtype.dtype
        if dtype == torch.float:
            return cls._global_float_value
        elif dtype == torch.double:
            return cls._global_double_value
        elif dtype == torch.half:
            return cls._global_half_value
        else:
            raise RuntimeError(f"Unsupported dtype for {cls.__name__}.")

    @classmethod
    def _set_value(cls, float_value, double_value, half_value):
        if float_value is not None:
            cls._global_float_value = float_value
        if double_value is not None:
            cls._global_double_value = double_value
        if half_value is not None:
            cls._global_half_value = half_value

    def __init__(self, float_value=None, double_value=None, half_value=None):
        self._orig_float_value = self.__class__.value(dtype=torch.float)
        self._instance_float_value = float_value
        self._orig_double_value = self.__class__.value(dtype=torch.double)
        self._instance_double_value = double_value
        self._orig_half_value = self.__class__.value(dtype=torch.half)
        self._instance_half_value = half_value

    def __enter__(
        self,
    ):
        self.__class__._set_value(
            self._instance_float_value,
            self._instance_double_value,
            self._instance_half_value,
        )

    def __exit__(self, *args):
        self.__class__._set_value(
            self._orig_float_value, self._orig_double_value, self._orig_half_value
        )
        return False


class _feature_flag:
    r"""Base class for feature flag settings with global scope.
    The default is set via the `_default` class attribute.
    """

    _default = False
    _state = None

    @classmethod
    def is_default(cls):
        return cls._state is None

    @classmethod
    def on(cls):
        if cls.is_default():
            return cls._default
        return cls._state

    @classmethod
    def off(cls):
        return not cls.on()

    @classmethod
    def _set_state(cls, state):
        cls._state = state

    def __init__(self, state=True):
        self.prev = self.__class__._state
        self.state = state

    def __enter__(self):
        self.__class__._set_state(self.state)

    def __exit__(self, *args):
        self.__class__._set_state(self.prev)
        return False


class _value_context:
    _global_value = None

    @classmethod
    def value(cls):
        return cls._global_value

    @classmethod
    def _set_value(cls, value):
        cls._global_value = value

    def __init__(self, value):
        self._orig_value = self.__class__.value()
        self._instance_value = value

    def __enter__(
        self,
    ):
        self.__class__._set_value(self._instance_value)

    def __exit__(self, *args):
        self.__class__._set_value(self._orig_value)
        return False


class debug(_feature_flag):
    """
    Whether or not to perform "safety" checks on the supplied data.
    (For example, that the correct training data is supplied in Exact GP training mode)
    Pros: fewer data checks, fewer warning messages
    Cons: possibility of supplying incorrect data, model accidentially in wrong mode

    (Default: True)
    """

    _default = True


class detach_test_caches(_feature_flag):
    """
    Whether or not to detach caches computed for making predictions. In most cases, you will want this,
    as this will speed up derivative computations of the predictions with respect to test inputs. However,
    if you also need derivatives with respect to training inputs (e.g., because you have fantasy observations),
    then you must disable this.

    (Default: True)
    """

    _default = True


class min_fixed_noise(_dtype_value_context):
    """
    The minimum noise value that can be used in :obj:`~gpytorch.likelihoods.FixedNoiseGaussianLikelihood`.
    If the supplied noise values are smaller than this, they are rounded up and a warning is raised.

    - Default for `float`: 1e-4
    - Default for `double`: 1e-6
    - Default for `half`: 1e-3
    """

    _global_float_value = 1e-4
    _global_double_value = 1e-6
    _global_half_value = 1e-3


class min_variance(_dtype_value_context):
    """
    The minimum variance that can be returned from :obj:`~gpytorch.distributions.MultivariateNormal#variance`.
    If variances are smaller than this, they are rounded up and a warning is raised.

    - Default for `float`: 1e-6
    - Default for `double`: 1e-10
    - Default for `half`: 1e-3
    """

    _global_float_value = 1e-6
    _global_double_value = 1e-10
    _global_half_value = 1e-3


class num_likelihood_samples(_value_context):
    """
    The number of samples to draw from a latent GP when computing a likelihood
    This is used in variational inference and training

    (Default: 10)
    """

    _global_value = 10


class num_gauss_hermite_locs(_value_context):
    """
    The number of samples to draw from a latent GP when computing a likelihood
    This is used in variational inference and training

    (Default: 20)
    """

    _global_value = 20


class cholesky_jitter(_dtype_value_context):
    """
    The jitter value used by `psd_safe_cholesky` when using cholesky solves.

    - Default for `float`: 1e-6
    - Default for `double`: 1e-8
    """

    _global_float_value = 1e-6
    _global_double_value = 1e-8


class cholesky_max_tries(_value_context):
    """
    The max_tries value used by `psd_safe_cholesky` when using cholesky solves.

    (Default: 3)
    """

    _global_value = 3

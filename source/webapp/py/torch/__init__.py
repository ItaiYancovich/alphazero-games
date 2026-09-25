"""Just enough of ``torch`` for the project's evaluators to run in a browser.

The web build has no PyTorch.  It does not need one: the networks run in
onnxruntime (see ``webgui.WebNet``), and what the evaluator classes do with
torch *around* the network is a handful of array operations -- wrap a numpy
array, move it to "cpu", mask the illegal moves, softmax, and read the numbers
back.  This module provides exactly those, on numpy, so every evaluator, agent
and search in the project runs unmodified.

Anything else a module touches at *import* time -- ``nn.Module`` as a base
class, ``@torch.no_grad()`` as a decorator, dtype names -- is here as an inert
stand-in.  Anything that would actually need PyTorch (building a layer,
tracing, loading a file) raises, and every caller that tries one already
catches the failure and falls back.
"""

from __future__ import annotations

import numpy as np

__version__ = "0.0-web-shim"


class _Unavailable(RuntimeError):
    pass


def _nope(*_a, **_k):
    raise _Unavailable("not available in the web build")


class Tensor:
    """A numpy array wearing the few methods the evaluators call."""

    __slots__ = ("a",)

    def __init__(self, a):
        self.a = np.asarray(a)

    # -- where the data lives (always here) --------------------------------
    def to(self, *_a, **_k):
        return self

    def cpu(self):
        return self

    def detach(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self.a

    def float(self):
        return Tensor(self.a.astype(np.float32))

    def item(self):
        return self.a.item()

    def tolist(self):
        return self.a.tolist()

    # -- shape --------------------------------------------------------------
    @property
    def shape(self):
        return self.a.shape

    def size(self, dim=None):
        return self.a.shape if dim is None else self.a.shape[dim]

    def dim(self):
        return self.a.ndim

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(self.a.reshape(shape))

    view = reshape

    def squeeze(self, dim=None):
        return Tensor(np.squeeze(self.a) if dim is None else np.squeeze(self.a, axis=dim))

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.a, dim))

    def __getitem__(self, key):
        return Tensor(self.a[key])

    def __len__(self):
        return len(self.a)

    def __array__(self, dtype=None, copy=None):
        return self.a if dtype is None else self.a.astype(dtype)

    # -- the arithmetic the evaluators do -----------------------------------
    def masked_fill(self, mask, value):
        out = self.a.copy()
        out[np.asarray(_unwrap(mask), dtype=bool)] = value
        return Tensor(out)

    def softmax(self, dim=-1):
        return softmax(self, dim=dim)

    def exp(self):
        return Tensor(np.exp(self.a))

    def sum(self, dim=None, keepdim=False):
        return Tensor(self.a.sum(axis=dim, keepdims=keepdim))

    def max(self, dim=None, keepdim=False):
        return Tensor(self.a.max(axis=dim, keepdims=keepdim))

    def __add__(self, o):
        return Tensor(self.a + _unwrap(o))

    __radd__ = __add__

    def __sub__(self, o):
        return Tensor(self.a - _unwrap(o))

    def __rsub__(self, o):
        return Tensor(_unwrap(o) - self.a)

    def __mul__(self, o):
        return Tensor(self.a * _unwrap(o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        return Tensor(self.a / _unwrap(o))

    def __neg__(self):
        return Tensor(-self.a)

    def __invert__(self):
        return Tensor(~self.a)

    def __repr__(self):
        return f"shim.Tensor({self.a!r})"


def _unwrap(x):
    return x.a if isinstance(x, Tensor) else x


def from_numpy(a):
    return Tensor(a)


def as_tensor(a, dtype=None, device=None):
    return Tensor(np.asarray(_unwrap(a)))


tensor = as_tensor


def softmax(x, dim=-1):
    a = _unwrap(x).astype(np.float64)
    a = a - a.max(axis=dim, keepdims=True)
    np.exp(a, out=a)
    a /= a.sum(axis=dim, keepdims=True)
    return Tensor(a.astype(np.float32))


def log_softmax(x, dim=-1):
    a = _unwrap(x).astype(np.float64)
    a = a - a.max(axis=dim, keepdims=True)
    a = a - np.log(np.exp(a).sum(axis=dim, keepdims=True))
    return Tensor(a.astype(np.float32))


def tanh(x):
    return Tensor(np.tanh(_unwrap(x)))


def sigmoid(x):
    return Tensor(1.0 / (1.0 + np.exp(-_unwrap(x))))


def cat(xs, dim=0):
    return Tensor(np.concatenate([_unwrap(x) for x in xs], axis=dim))


def stack(xs, dim=0):
    return Tensor(np.stack([_unwrap(x) for x in xs], axis=dim))


def zeros(*shape, **_k):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(np.zeros(shape, dtype=np.float32))


def load(path, *_a, **_k):
    """Only ever asked, in the app, what iteration a Splendor v3 file is.

    Real network loading is redirected to the ONNX models before anything can
    reach this, so a caller here is reading metadata: say "iteration 1".
    """
    return {"extra": {"iteration": 1}}


save = _nope


class _Mode:
    """``inference_mode()`` / ``no_grad()``: a context manager and a decorator."""

    def __init__(self, *_a, **_k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __call__(self, fn):
        return fn


inference_mode = no_grad = enable_grad = _Mode


def set_num_threads(*_a, **_k):
    pass


def get_num_threads():
    return 1


def manual_seed(*_a, **_k):
    pass


class device:  # noqa: N801 -- torch's own spelling
    def __init__(self, name="cpu"):
        self.type = str(name)

    def __repr__(self):
        return f"device({self.type!r})"


float32 = np.float32
float64 = np.float64
float16 = np.float16
int64 = long = np.int64
int32 = np.int32
uint8 = np.uint8
bool = np.bool_  # noqa: A001 -- torch's own name


class _Namespace:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)

    def __getattr__(self, name):
        return _nope


cuda = _Namespace(is_available=lambda: False, device_count=lambda: 0)
backends = _Namespace(mps=_Namespace(is_available=lambda: False))
jit = _Namespace(trace=_nope, script=_nope, optimize_for_inference=_nope, load=_nope)


def __getattr__(name):
    # Anything else is only ever referenced, never used, in the web build.
    return _nope

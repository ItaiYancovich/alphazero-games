"""``torch.nn`` for the web build: base classes to define networks against.

The project's network modules are imported (their ``load_checkpoint`` is what
the web build redirects to ONNX), so their class statements have to run -- but
no layer is ever constructed, because no network is ever built in Python.
"""

from __future__ import annotations

from .. import _nope


class Module:
    """Inert: subclassable, never meaningfully instantiated."""

    def __init__(self, *_a, **_k):
        pass

    def __call__(self, *a, **k):
        return self.forward(*a, **k)

    def forward(self, *_a, **_k):
        return _nope()

    def eval(self):
        return self

    def train(self, *_a):
        return self

    def to(self, *_a, **_k):
        return self

    def parameters(self):
        return iter(())


def _layer(name):
    class _Layer(Module):
        def __init__(self, *_a, **_k):
            _nope()
    _Layer.__name__ = name
    return _Layer


for _name in ("Conv1d", "Conv2d", "BatchNorm1d", "BatchNorm2d", "Linear", "ReLU",
              "GELU", "SiLU", "Tanh", "Sigmoid", "Dropout", "LayerNorm", "Embedding",
              "Sequential", "ModuleList", "ModuleDict", "Identity", "MultiheadAttention",
              "Flatten", "Softmax", "LogSoftmax", "AdaptiveAvgPool2d"):
    globals()[_name] = _layer(_name)


def Parameter(*_a, **_k):  # noqa: N802 -- torch's own name
    return _nope()


class _Init:
    def __getattr__(self, _name):
        return lambda *a, **k: None


init = _Init()


def __getattr__(name):
    return _layer(name)

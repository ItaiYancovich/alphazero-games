"""``torch.nn.functional``: referenced by network code, never called in the web build."""

from .. import _nope, log_softmax, sigmoid, softmax, tanh  # noqa: F401


def __getattr__(name):
    return _nope

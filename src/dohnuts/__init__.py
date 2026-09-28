"""Python tools for training, running, and evaluating Dohnuts models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dohnuts")
except PackageNotFoundError:
    __version__ = "0.1.0"

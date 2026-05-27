"""Col-Bandit: query-time top-K identification for late-interaction retrieval."""
from .core import ColBandit
from ._version import __version__

__all__ = ["ColBandit", "__version__"]

"""Col-Bandit: query-time top-K identification for late-interaction retrieval."""
from ._version import __version__

# Import the native kernel FIRST so `core.py`'s `from colbandit import _kernel`
# works during package init (otherwise core triggers a circular import on
# partially-initialised `colbandit`). Re-exporting the raw C entry points
# gives users direct, low-level access to the vendored CB-NK primitives
# (same calling convention as upstream NumKong).
try:
    from . import _kernel as _nk
    colbandit_flat = _nk.colbandit_flat
    topm_flat = _nk.topm_flat
    full_maxsim = _nk.full_maxsim
    maxsim_pack = _nk.maxsim_pack
    extract_flat_from_packed = _nk.extract_flat_from_packed
    total_tokens = _nk.total_tokens
except ImportError:  # pragma: no cover - matches core.py error handling
    pass

from .core import ColBandit

__all__ = [
    "ColBandit",
    "__version__",
    "colbandit_flat",
    "topm_flat",
    "full_maxsim",
    "maxsim_pack",
    "extract_flat_from_packed",
    "total_tokens",
]

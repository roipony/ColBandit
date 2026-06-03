"""Col-Bandit: query-time top-K identification for late-interaction retrieval."""
from .core import ColBandit
from ._version import __version__

# Re-export the raw C kernel entry points for users who want direct, low-level
# access to the vendored CB-NK primitives (same calling convention as upstream
# NumKong). The high-level ColBandit wrapper above is built on top of these.
try:
    from colbandit import _kernel as _nk
    colbandit_flat = _nk.colbandit_flat
    topm_flat = _nk.topm_flat
    full_maxsim = _nk.full_maxsim
    maxsim_pack = _nk.maxsim_pack
    extract_flat_from_packed = _nk.extract_flat_from_packed
    total_tokens = _nk.total_tokens
except ImportError:  # pragma: no cover - matches core.py error handling
    pass

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

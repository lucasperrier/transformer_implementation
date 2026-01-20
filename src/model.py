"""Legacy module.

The original contents of `src/model.py` were an incomplete draft.

The maintained implementation lives in `src/transformer.py` and is used by the
Lightning training loop in `src/lightning_module.py`.

This file re-exports `Transformer` to keep older imports working.
"""

from .transformer import Transformer  # noqa: F401

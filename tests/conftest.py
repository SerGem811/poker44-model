from __future__ import annotations

import sys
import types
from pathlib import Path

from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _Logging:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def trace(self, *args, **kwargs):
        pass

    def success(self, *args, **kwargs):
        pass

    def set_trace(self, *args, **kwargs):
        pass

    def set_config(self, *args, **kwargs):
        pass


class _Synapse(BaseModel):
    dendrite: object | None = None


fake_bittensor = types.SimpleNamespace(
    Synapse=_Synapse,
    logging=_Logging(),
    dendrite=object,
)

sys.modules.setdefault("bittensor", fake_bittensor)

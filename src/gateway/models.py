# -*- coding: utf-8 -*-
"""
A heterogeneous model group — stubs that stand in for real weights.

In production these hold, respectively, a 26B chat/reasoning model, a small
instruction model for cheap structured work, and a speech-to-text model — all
resident in one process's unified memory. Here they just sleep for a
representative, scaled-down time so the scheduler's behaviour is observable
without a GPU. The point is the *shape*: three different modalities, one lock,
one schedule.
"""
from __future__ import annotations

import time

from .core import Model

# Scaled-down stand-ins for measured latencies (real: chat ~0.5s, e4b ~0.8–2s,
# whisper ~0.6s). `load` in the payload scales it — a long "analyze" costs more
# than a short "chat" even on the same model.
_SCALE = 0.06


def _sleepy(base: float, text: str):
    def run(payload: dict) -> str:
        time.sleep(base * float(payload.get("load", 1.0)))
        return text.format(**{k: payload.get(k, "") for k in ("input",)})
    return run


def default_models() -> dict:
    return {
        # ~16–20GB in production; the fleet's chat + hard batch reasoning.
        "chat_26b": Model("chat_26b", latency_s=_SCALE * 9,
                          run=_sleepy(_SCALE * 9, "chat_26b: {input}")),
        # ~3GB; fast, format-forced classification & extraction (80–90% of batch work).
        "classify_e4b": Model("classify_e4b", latency_s=_SCALE * 3,
                             run=_sleepy(_SCALE * 3, 'classify_e4b: {{"label": "{input}"}}')),
        # ~3GB; speech-to-text.
        "whisper": Model("whisper", latency_s=_SCALE * 4,
                        run=_sleepy(_SCALE * 4, "whisper: transcript of {input}")),
    }

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Worked example. Run:

    python examples/demo.py

Fires a mixed fleet at one GPU: a batch backlog (analyze/summarize) plus
interactive chat and speech arriving in the middle. Shows three things:

  1. the GPU never runs two jobs at once (max_inflight stays 1),
  2. interactive work jumps the batch backlog (priority),
  3. the batch backlog still drains and nobody starves (aging), and a lease lets
     a bulk job burst without freezing the interactive lane.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gateway import Gateway, default_models  # noqa: E402


def scenario(title, use_lease):
    print(f"\n=== {title} ===")
    gw = Gateway(default_models()).start()

    # a batch backlog lands first (a pipeline dumped 6 jobs)
    jobs = []
    for i in range(6):
        kind = "analyze" if i % 2 else "summarize"
        jobs.append(gw.submit(kind, {"input": f"doc{i}", "load": 1.4 if kind == "analyze" else 0.6}))

    if use_lease:
        gw.acquire_batch_lease(boost=8.0, ttl_s=5.0)

    # ...then a person shows up mid-backlog: chat + a voice note
    time.sleep(0.03)
    jobs.append(gw.submit("chat", {"input": "hello", "load": 1.0}))
    jobs.append(gw.submit("transcribe", {"input": "voice.wav", "load": 1.0}))
    time.sleep(0.03)
    jobs.append(gw.submit("chat", {"input": "follow-up", "load": 1.0}))

    for j in jobs:
        j.wait(timeout=10)
    gw.stop()

    # print dispatch order by when they finished (reconstruct from waits)
    finished = sorted(jobs, key=lambda j: j.submit_t + j.lock_wait_ms / 1000 + j.infer_ms / 1000)
    print("  finish │ kind        queue        waited  infer")
    for j in finished:
        tag = "⚡" if j.priority == 0 else "  "
        print(f"   {tag}    │ {j.kind:<11} {j.queue:<12} {j.lock_wait_ms:>4}ms {j.infer_ms:>4}ms")
    rep = gw.report()
    print(f"  max_inflight = {rep['max_inflight']}   (must be 1 — the safety guarantee)")
    chat_wait = rep["kinds"].get("chat", {}).get("avg_wait_ms", 0)
    print(f"  avg chat wait = {chat_wait}ms   (interactive stays responsive under batch load)")


if __name__ == "__main__":
    scenario("strict priority — interactive jumps the batch backlog", use_lease=False)
    scenario("with a batch lease — bulk job bursts, chat still gets in", use_lease=True)

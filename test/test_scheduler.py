# -*- coding: utf-8 -*-
"""
The guarantees, as tests. Run: `python -m pytest test/` or `python test/test_scheduler.py`.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gateway import DEFAULT_ROUTES, Gateway, Model, Priority, Route, Router  # noqa: E402


def _fast_models(latency=0.01):
    # small *real* latency so a backlog actually exists for interactive work to jump
    def mk(key):
        def run(p, k=key):
            time.sleep(latency)
            return k
        return Model(key, latency, run)
    return {"chat_26b": mk("chat_26b"), "classify_e4b": mk("classify_e4b"), "whisper": mk("whisper")}


# ── the safety invariant: never two jobs on the GPU at once ──────────────────
def test_mutual_exclusion_under_concurrent_submit():
    # a model that would *notice* overlap: bump a shared counter, assert it's 1.
    live = {"n": 0, "max": 0}
    guard = threading.Lock()

    def watch(payload):
        with guard:
            live["n"] += 1
            live["max"] = max(live["max"], live["n"])
        time.sleep(0.01)
        with guard:
            live["n"] -= 1
        return "ok"

    models = {"chat_26b": Model("chat_26b", 0.01, watch),
              "classify_e4b": Model("classify_e4b", 0.01, watch),
              "whisper": Model("whisper", 0.01, watch)}
    gw = Gateway(models).start()
    # 20 clients slam it at once, across kinds/models
    kinds = ["chat", "classify", "transcribe", "summarize", "analyze"]
    jobs = [gw.submit(kinds[i % len(kinds)], {"input": i}) for i in range(20)]
    for j in jobs:
        j.wait(timeout=10)
    gw.stop()
    assert live["max"] == 1, f"GPU ran {live['max']} jobs concurrently"
    assert gw.max_inflight == 1


# ── priority: interactive jumps a batch backlog ──────────────────────────────
def test_interactive_beats_batch_backlog():
    gw = Gateway(_fast_models(0.01), aging_s=1e6).start()  # disable aging to isolate priority
    # queue a big batch backlog first
    batch = [gw.submit("analyze", {"input": i}) for i in range(8)]
    time.sleep(0.002)
    # one interactive job arrives after the backlog — it should still finish early
    chat = gw.submit("chat", {"input": "urgent"})

    for j in batch + [chat]:
        j.wait(timeout=10)
    gw.stop()
    order = sorted(batch + [chat], key=lambda j: (j.submit_t + j.lock_wait_ms / 1000 + j.infer_ms / 1000))
    chat_pos = order.index(chat)
    assert chat_pos <= 2, f"interactive chat finished at position {chat_pos}, expected near front"


# ── anti-starvation: the scoring math, tested directly (deterministic) ────────
def test_aging_lets_a_starving_low_job_overtake_fresh_high():
    from src.gateway.core import Job, Priority as P
    gw = Gateway(_fast_models(), aging_s=0.5)   # not started; we call _score directly
    now = 100.0

    def job(priority, submitted_ago):
        return Job(kind="k", payload={}, model_key="chat_26b", queue="q",
                   priority=priority, seq=1, submit_t=now - submitted_ago)

    fresh_high = job(P.HIGH, 0.0)
    fresh_low = job(P.LOW, 0.0)
    # fresh: HIGH is served before LOW (lower score = sooner)
    assert gw._score(fresh_high, now) < gw._score(fresh_low, now)
    # a LOW that has waited past the priority gap overtakes a fresh HIGH → no
    # permanent starvation. The gap is (LOW - HIGH) aging-units of waiting.
    gap = int(P.LOW) - int(P.HIGH)
    starving_low = job(P.LOW, submitted_ago=(gap + 1) * gw.aging_s)
    assert gw._score(starving_low, now) < gw._score(fresh_high, now)


def test_batch_lease_boosts_low_priority():
    from src.gateway.core import Job, Priority as P
    gw = Gateway(_fast_models(), aging_s=1e9)   # aging off, isolate the lease effect
    now = 100.0

    def low(): return Job("k", {}, "chat_26b", "q", P.LOW, 1, now)
    def high(): return Job("k", {}, "chat_26b", "q", P.HIGH, 1, now)

    assert gw._score(low(), now) > gw._score(high(), now)   # without lease, HIGH wins
    gw.acquire_batch_lease(boost=20.0, ttl_s=1000.0)
    gw._lease.expires_t = now + 1000.0                       # (monotonic-independent for the test)
    assert gw._score(low(), now) < gw._score(high(), now)   # lease flips it: batch bursts


# ── routing: a kind resolves to its model, queue, priority ───────────────────
def test_routing_and_unknown_kind():
    gw = Gateway(_fast_models())
    assert gw.router.resolve("chat").model_key == "chat_26b"
    assert gw.router.resolve("summarize").priority == Priority.LOW
    assert gw.router.resolve("analyze").queue == "batch"
    try:
        gw.router.resolve("nonsense")
        assert False, "unknown kind should raise"
    except KeyError:
        pass


def test_custom_route_repoints_without_touching_caller():
    # re-point 'summarize' from the small model to the big one; caller code identical
    routes = dict(DEFAULT_ROUTES)
    routes["summarize"] = Route("chat_26b", "batch", Priority.LOW)
    gw = Gateway(_fast_models(), router=Router(routes)).start()
    j = gw.call("summarize", {"input": "x"}, timeout=10)
    gw.stop()
    assert j.result == "chat_26b"   # served by the re-pointed model, caller unchanged


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")

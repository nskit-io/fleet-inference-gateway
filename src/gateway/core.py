# -*- coding: utf-8 -*-
"""
Fleet Inference Gateway — the engine.

One process owns the GPU. Everyone else is a thin client.

On Apple Silicon you cannot run two inference processes against the Metal GPU at
once — they race on the command buffer and crash — and there is no MPS/MIG to
partition it, nor memory isolation under unified memory. So the only safe design
is: a *single* process holds every model, serializes all GPU work behind one
in-process lock, and answers callers over HTTP. This module is that design,
distilled and dependency-free so it runs anywhere (no GPU, no MLX) and you can
read it in one sitting.

Three ideas beyond "a local model server":

  1. Address the GPU by *role* (a `kind`), not by model name. The caller asks for
     `summarize` or `transcribe`; the router decides which model, which queue, and
     what priority. Callers stay dumb; you re-route a kind without touching them.

  2. One *heterogeneous* group behind the lock — a big chat model, a small
     classifier, and a speech model share the same GPU and the same schedule.

  3. Schedule a *mixed fleet*: keep interactive work (chat) in its own queue so a
     stalled or flooded batch queue can never block it, and let batch bursts make
     progress via a lease without starving the interactive lane.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Models — a heterogeneous group. In production these hold real weights; here they
# are stubs that sleep for a representative time so the scheduler is observable.

class Model:
    """A loaded model. `run` is the only thing that touches the GPU."""

    def __init__(self, key: str, latency_s: float, run: Callable[[dict], str]):
        self.key = key
        self.latency_s = latency_s
        self._run = run

    def infer(self, payload: dict) -> str:
        return self._run(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Jobs

class Priority(IntEnum):
    HIGH = 0     # interactive — a person is waiting
    LOW = 10     # batch — a pipeline is waiting


@dataclass(order=False)
class Job:
    kind: str
    payload: dict
    model_key: str
    queue: str
    priority: Priority
    seq: int
    submit_t: float
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[str] = None
    error: Optional[str] = None
    lock_wait_ms: int = 0
    infer_ms: int = 0

    def wait(self, timeout: Optional[float] = None) -> "Job":
        self.done.wait(timeout)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Router — kind → (model, queue, priority). This table is the whole contract a
# caller sees. Re-point a kind here and every client follows, untouched.

@dataclass
class Route:
    model_key: str
    queue: str
    priority: Priority


class Router:
    def __init__(self, routes: Dict[str, Route]):
        self.routes = dict(routes)

    def resolve(self, kind: str) -> Route:
        if kind not in self.routes:
            raise KeyError(f"unknown kind {kind!r}; known: {sorted(self.routes)}")
        return self.routes[kind]


# Default routes, mirroring a real deployment: a 26B chat model for interactive and
# for hard batch reasoning, a small E4B model for cheap structured batch work, and
# a speech model. Note two *kinds* share the 26B model but live in different queues
# and priorities — because the queue, not the model, is what isolates the fleet.
DEFAULT_ROUTES: Dict[str, Route] = {
    "chat":       Route("chat_26b", "interactive", Priority.HIGH),
    "classify":   Route("classify_e4b", "interactive", Priority.HIGH),
    "transcribe": Route("whisper", "interactive", Priority.HIGH),
    "summarize":  Route("classify_e4b", "batch", Priority.LOW),
    "entities":   Route("classify_e4b", "batch", Priority.LOW),
    "analyze":    Route("chat_26b", "batch", Priority.LOW),   # hard reasoning, but batch
    "fact":       Route("chat_26b", "batch", Priority.LOW),
}


# ─────────────────────────────────────────────────────────────────────────────
# Gateway — the single-owner GPU + the scheduler.

@dataclass
class Lease:
    """A batch-priority lease: while held, batch jobs get an aging boost so a big
    backfill makes steady progress without fully starving interactive work."""
    boost: float
    expires_t: float


class Gateway:
    def __init__(self, models: Dict[str, Model], router: Optional[Router] = None,
                 aging_s: float = 2.0):
        self.models = dict(models)
        self.router = router or Router(DEFAULT_ROUTES)
        self.aging_s = aging_s              # a LOW job reaches HIGH after ~aging_s * (LOW-HIGH)/… of waiting

        self._pending: List[Job] = []
        self._cv = threading.Condition()
        self._seq = 0
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._lease: Optional[Lease] = None

        # observability — mirrors /stats and /admin/queue-stats
        self._inflight = 0
        self.max_inflight = 0               # MUST stay 1: the core safety guarantee
        self.stats: Dict[str, dict] = {}

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "Gateway":
        self._running = True
        self._worker = threading.Thread(target=self._loop, name="gpu", daemon=True)
        self._worker.start()
        return self

    def stop(self) -> None:
        with self._cv:
            self._running = False
            self._cv.notify_all()
        if self._worker:
            self._worker.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    # -- submit (what a thin client calls) ---------------------------------
    def submit(self, kind: str, payload: dict) -> Job:
        route = self.router.resolve(kind)
        if route.model_key not in self.models:
            raise KeyError(f"kind {kind!r} routes to missing model {route.model_key!r}")
        with self._cv:
            self._seq += 1
            job = Job(kind=kind, payload=payload, model_key=route.model_key,
                      queue=route.queue, priority=route.priority,
                      seq=self._seq, submit_t=time.monotonic())
            self._pending.append(job)
            st = self.stats.setdefault(kind, {"submitted": 0, "completed": 0, "wait_ms_total": 0})
            st["submitted"] += 1
            self._cv.notify()
        return job

    def call(self, kind: str, payload: dict, timeout: Optional[float] = None) -> Job:
        """Convenience: submit and block for the result (the synchronous HTTP call)."""
        return self.submit(kind, payload).wait(timeout)

    # -- batch lease (v0.7 /admin/mode/acquire) ----------------------------
    def acquire_batch_lease(self, boost: float = 8.0, ttl_s: float = 30.0) -> None:
        with self._cv:
            self._lease = Lease(boost=boost, expires_t=time.monotonic() + ttl_s)
            self._cv.notify()

    def release_batch_lease(self) -> None:
        with self._cv:
            self._lease = None

    # -- scheduler ---------------------------------------------------------
    def _score(self, job: Job, now: float) -> float:
        """Lower = served sooner. Base priority, minus an aging bonus so a starving
        LOW job eventually outranks fresh HIGH work; a batch lease adds more bonus.
        This single expression gives priority AND anti-starvation together."""
        waited = now - job.submit_t
        bonus = waited / self.aging_s
        if job.priority == Priority.LOW and self._lease and now < self._lease.expires_t:
            bonus += self._lease.boost
        return float(job.priority) - bonus

    def _select(self, now: float) -> Optional[int]:
        if not self._pending:
            return None
        best_i, best_score = 0, self._score(self._pending[0], now)
        for i in range(1, len(self._pending)):
            s = self._score(self._pending[i], now)
            # tie-break by submission order (FIFO within equal effective priority)
            if s < best_score or (s == best_score and self._pending[i].seq < self._pending[best_i].seq):
                best_i, best_score = i, s
        return best_i

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._running and not self._pending:
                    self._cv.wait()
                if not self._running and not self._pending:
                    return
                i = self._select(time.monotonic())
                job = self._pending.pop(i)

            # --- the GPU section. Single worker → mutual exclusion by construction.
            # The counter is a live assertion of the guarantee, surfaced as max_inflight.
            self._inflight += 1
            self.max_inflight = max(self.max_inflight, self._inflight)
            start = time.monotonic()
            job.lock_wait_ms = int((start - job.submit_t) * 1000)
            try:
                job.result = self.models[job.model_key].infer(job.payload)
            except Exception as e:  # a bad job must not take down the worker
                job.error = repr(e)
            job.infer_ms = int((time.monotonic() - start) * 1000)
            self._inflight -= 1

            st = self.stats[job.kind]
            st["completed"] += 1
            st["wait_ms_total"] += job.lock_wait_ms
            job.done.set()

    # -- reporting (/stats) ------------------------------------------------
    def report(self) -> dict:
        return {
            "max_inflight": self.max_inflight,   # invariant: 1
            "kinds": {
                k: {**v, "avg_wait_ms": (v["wait_ms_total"] // v["completed"]) if v["completed"] else 0}
                for k, v in sorted(self.stats.items())
            },
        }

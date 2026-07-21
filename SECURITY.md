# Security Policy

This repository is an architecture pattern and a dependency-free reference
implementation. The reference has no network listener, no third-party
dependencies, and ships stub models — its attack surface is small. Still, if you
find a problem, we want to hear about it.

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Email
**nskit@nskit.io** with a description, reproduction steps, and the commit you
observed it on. We aim to acknowledge within a few business days.

## Scope

In scope:

- a scheduling input that breaks the core safety invariant — two jobs executing
  on the single worker at once (`max_inflight > 1`);
- a crafted submission that makes the scheduler deadlock, starve a queue without
  bound, or crash the worker thread.

Out of scope:

- the *policy* of any default route or priority (open a normal issue/PR);
- production concerns of a real deployment (auth on the HTTP layer, TLS, rate
  limiting) — those belong to the server you build around this pattern, not the
  reference.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The economics. Run:

    python examples/economics.py

The gateway's real payoff is not latency — it's the bill. One owned Apple-Silicon
machine has a (near) flat cost: hardware amortized over its life, plus power. A
metered cloud API scales with every token. Below some volume the cloud is cheaper;
above it, the owned machine wins and keeps winning — and a fleet of apps sharing
one machine crosses that line far sooner than any single app would.

This is a back-of-envelope model, not a benchmark. Numbers are inputs — change them
to your own. The shape is the point.
"""


def cloud_monthly_usd(reqs_per_day, in_tok, out_tok, price_in_per_m, price_out_per_m):
    daily = reqs_per_day * (in_tok * price_in_per_m + out_tok * price_out_per_m) / 1_000_000
    return daily * 30.0


def owned_monthly_usd(machine_usd, life_months, watts, power_price_kwh):
    amortize = machine_usd / life_months
    power = watts / 1000.0 * 24 * 30 * power_price_kwh
    return amortize + power


def breakeven_reqs_per_day(owned_month, in_tok, out_tok, price_in_per_m, price_out_per_m):
    per_req = (in_tok * price_in_per_m + out_tok * price_out_per_m) / 1_000_000
    return owned_month / 30.0 / per_req if per_req else float("inf")


if __name__ == "__main__":
    # ---- inputs (edit to your own) --------------------------------------
    MACHINE_USD, LIFE_MONTHS, WATTS, POWER = 5_600, 36, 120, 0.16   # a Mac Studio, 3-yr life
    PRICE_IN, PRICE_OUT = 0.60, 2.40                                # $/1M tokens (a mid GPT-class API)
    IN_TOK, OUT_TOK = 1_500, 400                                    # avg per request

    owned = owned_monthly_usd(MACHINE_USD, LIFE_MONTHS, WATTS, POWER)
    be = breakeven_reqs_per_day(owned, IN_TOK, OUT_TOK, PRICE_IN, PRICE_OUT)

    print("Owned Apple-Silicon machine")
    print(f"  ${MACHINE_USD} / {LIFE_MONTHS}mo + {WATTS}W  →  ${owned:,.0f} / month, flat\n")
    print(f"Break-even ≈ {be:,.0f} requests/day  (below: cloud wins; above: the machine wins)\n")

    print(f"  {'reqs/day':>10} │ {'cloud/mo':>10} │ {'owned/mo':>9} │ verdict")
    print(f"  {'-'*10}─┼─{'-'*10}─┼─{'-'*9}─┼─────────")
    for r in (1_000, 5_000, 20_000, 50_000, 150_000, 500_000):
        cloud = cloud_monthly_usd(r, IN_TOK, OUT_TOK, PRICE_IN, PRICE_OUT)
        win = "owned ✓" if cloud > owned else "cloud ✓"
        mult = f"  ({cloud/owned:.1f}× cloud)" if cloud > owned else ""
        print(f"  {r:>10,} │ ${cloud:>8,.0f} │ ${owned:>7,.0f} │ {win}{mult}")

    print("\n  A fleet of ~30 apps sharing one machine reaches fleet-wide volume that no")
    print("  single app would — so the machine is past break-even for the *portfolio*")
    print("  long before it would be for any one app. That is the fleet economics.")

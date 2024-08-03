"""The 60-second artefact: the same money, a different rota.

Run: python -m src.demo
"""

from __future__ import annotations

from .sim import (
    BREACH_HOURS, demand_shaped_roster, flat_roster, simulate,
)
from .validate import MMc

WEEKS = 12
ARRIVALS = 7.0
SEED = 17


def row(label: str, r) -> str:
    return (
        f"  {label:<26}{r.mean_wait:>8.2f}{r.percentile(50):>8.2f}"
        f"{r.percentile(95):>8.2f}{r.percentile(99):>8.2f}"
        f"{r.breach_rate:>10.1%}{r.lwbs_rate:>9.1%}"
    )


def main() -> None:
    print("\n  SURGECAPACITY - why staffing to the average always fails")
    print("  " + "=" * 74)

    analytic = MMc(4.0, 1 / 1.2, 6)
    print("  First, the simulator is validated against closed-form M/M/c:")
    print(f"    utilisation {analytic.utilisation:.0%}   "
          f"mean wait {analytic.mean_wait:.2f}h   "
          f"p95 wait {analytic.wait_percentile(95):.2f}h")
    print(f"    the p95 is {analytic.wait_percentile(95) / analytic.mean_wait:.1f}x "
          "the mean - that ratio is the entire problem.\n")

    flat = flat_roster(9)
    shaped = demand_shaped_roster(flat.total_hours)
    moved = flat.moved(from_hour=1 * 24 + 3, to_hour=5 * 24 + 20, count=2)

    a = simulate(flat, weeks=WEEKS, arrivals_per_hour_mean=ARRIVALS, seed=SEED)
    b = simulate(shaped, weeks=WEEKS, arrivals_per_hour_mean=ARRIVALS, seed=SEED)
    c = simulate(moved, weeks=WEEKS, arrivals_per_hour_mean=ARRIVALS, seed=SEED)

    print(f"  {WEEKS} weeks, {a.arrivals:,} attendances, "
          f"{flat.total_hours:,} staffed hours per week in every scenario.\n")
    print(f"  {'roster':<26}{'mean':>8}{'p50':>8}{'p95':>8}{'p99':>8}"
          f"{'breach':>10}{'LWBS':>9}")
    print("  " + "-" * 74)
    print(row("flat (staffed to mean)", a))
    print(row("demand-shaped", b))
    print(row("flat, 2 moved Tue3am->Sat8pm", c))

    print(f"\n  Every row costs exactly the same: {flat.total_hours:,} clinician-hours.")
    print(f"  The {BREACH_HOURS:.0f}-hour breach rate moves from {a.breach_rate:.1%} "
          f"to {b.breach_rate:.1%}.\n")

    mean_saved = (a.mean_wait - b.mean_wait) * 60
    p95_saved = (a.percentile(95) - b.percentile(95)) * 60
    print("  Reported as an average, the change looks modest:")
    print(f"    mean wait falls {mean_saved:>6.0f} minutes")
    print(f"    p95 wait falls  {p95_saved:>6.0f} minutes")
    print("  The tail is where the hours actually are, and the tail is what the")
    print("  target, the ambulance handover and the patient all experience.\n")

    surge = simulate(flat, weeks=WEEKS, arrivals_per_hour_mean=ARRIVALS,
                     seed=SEED, surge_multiplier=1.25)
    print("  A 25% surge (flu peak) against the flat roster:")
    print(f"    mean wait {a.mean_wait:.2f}h -> {surge.mean_wait:.2f}h "
          f"(+{(surge.mean_wait - a.mean_wait) * 60:.0f} min)")
    print(f"    p95 wait  {a.percentile(95):.2f}h -> {surge.percentile(95):.2f}h "
          f"(+{(surge.percentile(95) - a.percentile(95)) * 60:.0f} min)")
    print(f"    breach    {a.breach_rate:.1%} -> {surge.breach_rate:.1%}")
    print("  A 25% increase in demand does not cost 25% more waiting. That")
    print("  non-linearity is why capacity planning on averages fails, and")
    print("  why it fails suddenly rather than gradually.\n")


if __name__ == "__main__":
    main()

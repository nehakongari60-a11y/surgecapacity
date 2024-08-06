"""Discrete-event simulation of an emergency department.

The core model: tail-first reporting and hour-of-week arrivals.

An ED staffed to average demand is understaffed roughly half the time. Because
queues are non-linear, that half generates almost all of the twelve-hour waits,
the ambulance diverts and the patients who leave without being seen. Staffing to
the mean is a mathematically guaranteed failure that looks entirely reasonable
on a spreadsheet, which is why it survives review.

So this simulation reports the p95 wait and the breach probability, not the
mean. In any queueing system approaching capacity the mean understates the tail
by an order of magnitude, and the tail is what the target is written against.

It also models arrivals by hour-of-week rather than as a flat rate. A model with
a flat arrival rate recommends Tuesday's staffing for Saturday night, which is
precisely the mistake the department is already making.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

HOURS_PER_WEEK = 168


# ---------------------------------------------------------------------------
# Arrivals
# ---------------------------------------------------------------------------

#: Relative arrival intensity by hour of day, shaped like published A&E
#: attendance curves: a morning ramp, an afternoon peak, a small evening bump.
HOUR_OF_DAY_PROFILE: tuple[float, ...] = (
    0.45, 0.35, 0.30, 0.28, 0.30, 0.40, 0.65, 0.95,
    1.25, 1.45, 1.50, 1.45, 1.40, 1.35, 1.30, 1.30,
    1.35, 1.40, 1.35, 1.20, 1.05, 0.90, 0.75, 0.58,
)

#: Day-of-week multipliers. Monday is index 0. Weekends are heavier and, more
#: importantly, heavier at different HOURS.
DAY_OF_WEEK_PROFILE: tuple[float, ...] = (
    1.12, 1.02, 0.98, 0.97, 1.05, 1.18, 1.20,
)


def intensity(hour_of_week: int) -> float:
    day = (hour_of_week // 24) % 7
    hour = hour_of_week % 24
    return HOUR_OF_DAY_PROFILE[hour] * DAY_OF_WEEK_PROFILE[day]


#: Triage categories: (label, share, mean service hours, priority)
#: Lower priority number is seen first.
TRIAGE = (
    ("resus", 0.02, 2.50, 0),
    ("very_urgent", 0.10, 1.80, 1),
    ("urgent", 0.33, 1.10, 2),
    ("standard", 0.42, 0.70, 3),
    ("non_urgent", 0.13, 0.45, 4),
)

#: Four-hour target, the operational definition of a breach.
BREACH_HOURS = 4.0


@dataclass(frozen=True)
class Patient:
    id: int
    arrival: float
    category: str
    priority: int
    service_hours: float


@dataclass
class Result:
    waits: list[float] = field(default_factory=list)
    waits_by_category: dict[str, list[float]] = field(default_factory=dict)
    waits_by_hour_of_week: dict[int, list[float]] = field(default_factory=dict)
    left_without_being_seen: int = 0
    arrivals: int = 0
    #: Total clinician-hours the scenario paid for.
    staffed_hours: float = 0.0

    def percentile(self, p: float) -> float:
        if not self.waits:
            return 0.0
        s = sorted(self.waits)
        idx = min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))
        return s[idx]

    @property
    def mean_wait(self) -> float:
        return sum(self.waits) / len(self.waits) if self.waits else 0.0

    @property
    def breach_rate(self) -> float:
        if not self.waits:
            return 0.0
        return sum(1 for w in self.waits if w > BREACH_HOURS) / len(self.waits)

    @property
    def lwbs_rate(self) -> float:
        return self.left_without_being_seen / self.arrivals if self.arrivals else 0.0


# ---------------------------------------------------------------------------
# Staffing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Roster:
    """Clinicians on duty for each hour of the week."""

    name: str
    by_hour: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.by_hour) != HOURS_PER_WEEK:
            raise ValueError("a roster must cover all 168 hours of the week")

    def at(self, sim_hours: float) -> int:
        return self.by_hour[int(sim_hours) % HOURS_PER_WEEK]

    @property
    def total_hours(self) -> int:
        return sum(self.by_hour)

    def moved(self, from_hour: int, to_hour: int, count: int) -> Roster:
        """Move `count` clinicians from one hour of the week to another.

        Cost-neutral by construction, which is what makes the resulting
        comparison a scheduling argument rather than a budget request.
        """
        hours = list(self.by_hour)
        hours[from_hour % HOURS_PER_WEEK] -= count
        hours[to_hour % HOURS_PER_WEEK] += count
        return Roster(f"{self.name}+moved", tuple(hours))


def flat_roster(clinicians: int, name: str = "flat") -> Roster:
    """The staffing a mean-based capacity model produces."""
    return Roster(name, tuple([clinicians] * HOURS_PER_WEEK))


def demand_shaped_roster(total_hours: int, name: str = "demand-shaped") -> Roster:
    """The same total staffed hours, distributed in proportion to demand."""
    weights = [intensity(h) for h in range(HOURS_PER_WEEK)]
    scale = total_hours / sum(weights)
    raw = [w * scale for w in weights]
    hours = [max(1, int(x)) for x in raw]

    # Hand back any rounding shortfall to the busiest hours.
    shortfall = total_hours - sum(hours)
    order = sorted(range(HOURS_PER_WEEK), key=lambda h: -weights[h])
    i = 0
    while shortfall > 0:
        hours[order[i % HOURS_PER_WEEK]] += 1
        shortfall -= 1
        i += 1
    while shortfall < 0:
        h = order[-1 - (i % HOURS_PER_WEEK)]
        if hours[h] > 1:
            hours[h] -= 1
            shortfall += 1
        i += 1
    return Roster(name, tuple(hours))


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------

def simulate(
    roster: Roster,
    weeks: int = 8,
    arrivals_per_hour_mean: float = 8.0,
    seed: int = 1,
    balk_after_hours: float = 6.0,
    surge_multiplier: float = 1.0,
) -> Result:
    """Priority-queue simulation with hour-of-week arrival intensity.

    Service is exponential; arrivals are a non-homogeneous Poisson process
    thinned against the hour-of-week profile.
    """
    rnd = random.Random(seed)
    horizon = weeks * HOURS_PER_WEEK
    result = Result()
    result.staffed_hours = roster.total_hours * weeks

    # --- generate arrivals -------------------------------------------------
    peak_rate = arrivals_per_hour_mean * max(
        intensity(h) for h in range(HOURS_PER_WEEK)) * surge_multiplier
    patients: list[Patient] = []
    t = 0.0
    pid = 0
    while t < horizon:
        t += rnd.expovariate(peak_rate)
        if t >= horizon:
            break
        # Thinning: accept with probability proportional to this hour's rate.
        accept = (
            arrivals_per_hour_mean * intensity(int(t)) * surge_multiplier
        ) / peak_rate
        if rnd.random() > accept:
            continue
        roll = rnd.random()
        cumulative = 0.0
        for label, share, mean_service, priority in TRIAGE:
            cumulative += share
            if roll <= cumulative:
                pid += 1
                patients.append(Patient(
                    pid, t, label, priority, rnd.expovariate(1 / mean_service)))
                break
    result.arrivals = len(patients)

    # --- run ---------------------------------------------------------------
    # A heap of (free_at) per server slot is not enough, because capacity
    # changes hourly. Instead: track busy-until times and the number on duty.
    busy: list[float] = []          # heap of completion times
    waiting: list[tuple[int, float, Patient]] = []  # heap by (priority, arrival)
    now = 0.0
    index = 0

    def release(until: float) -> None:
        while busy and busy[0] <= until:
            heapq.heappop(busy)

    while index < len(patients) or waiting:
        if index < len(patients) and (not waiting or patients[index].arrival <= now):
            p = patients[index]
            index += 1
            now = max(now, p.arrival)
            heapq.heappush(waiting, (p.priority, p.arrival, p))

        release(now)
        on_duty = roster.at(now)

        if waiting and len(busy) < on_duty:
            _, _, p = heapq.heappop(waiting)
            wait = max(0.0, now - p.arrival)

            if wait >= balk_after_hours and p.priority >= 3:
                # Lower-acuity patients leave. Higher-acuity ones do not, which
                # is why the tail matters so much for them.
                result.left_without_being_seen += 1
                continue

            result.waits.append(wait)
            result.waits_by_category.setdefault(p.category, []).append(wait)
            result.waits_by_hour_of_week.setdefault(
                int(p.arrival) % HOURS_PER_WEEK, []).append(wait)
            heapq.heappush(busy, now + p.service_hours)
            continue

        # Nothing can start now: advance to the next event.
        candidates: list[float] = []
        if busy:
            candidates.append(busy[0])
        if index < len(patients):
            candidates.append(patients[index].arrival)
        # Capacity may increase at the next hour boundary.
        candidates.append(math.floor(now) + 1.0)
        nxt = min(candidates)
        if nxt <= now:
            nxt = now + 1e-6
        now = nxt

    return result

"""Analytic M/M/c, used to validate the simulator before trusting it.

A simulator nobody has validated produces confident nonsense, and the output
looks identical either way. M/M/c has a closed form, so the simulator can be
pointed at a case where the right answer is already known. Everything the
project claims rests on this check passing first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def erlang_c(servers: int, offered_load: float) -> float:
    """Probability an arriving customer has to wait at all."""
    c, a = servers, offered_load
    if a >= c:
        return 1.0
    total = sum((a ** k) / math.factorial(k) for k in range(c))
    last = (a ** c) / (math.factorial(c) * (1 - a / c))
    return last / (total + last)


@dataclass(frozen=True)
class MMc:
    arrival_rate: float     # lambda
    service_rate: float     # mu
    servers: int            # c

    @property
    def offered_load(self) -> float:
        return self.arrival_rate / self.service_rate

    @property
    def utilisation(self) -> float:
        return self.offered_load / self.servers

    @property
    def prob_wait(self) -> float:
        return erlang_c(self.servers, self.offered_load)

    @property
    def mean_wait(self) -> float:
        """Wq: expected time in queue (not in system)."""
        c, mu = self.servers, self.service_rate
        return self.prob_wait / (c * mu - self.arrival_rate)

    def wait_percentile(self, p: float) -> float:
        """Analytic quantile of the waiting time.

        P(W > t) = C * exp(-(c*mu - lambda) * t), so the quantile inverts
        directly. Including this - rather than only the mean - is the point:
        the whole argument of the project is that the tail behaves very
        differently from the average, and here it can be checked exactly.
        """
        target = 1 - p / 100
        c, mu, lam = self.servers, self.service_rate, self.arrival_rate
        cw = self.prob_wait
        if target >= cw:
            return 0.0
        return -math.log(target / cw) / (c * mu - lam)

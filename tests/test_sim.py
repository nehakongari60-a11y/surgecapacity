"""Validate against closed form first. Then, and only then, believe the model."""

from __future__ import annotations

import statistics
import unittest

from src.sim import (
    BREACH_HOURS, HOURS_PER_WEEK, Roster, demand_shaped_roster, flat_roster,
    intensity, simulate,
)
from src.validate import MMc, erlang_c


def mmc_sim(servers: int, arrival_rate: float, mean_service: float,
            weeks: int, seed: int):
    """Run the simulator in an M/M/c configuration: flat arrivals, one class."""
    import src.sim as sim_module

    original_triage = sim_module.TRIAGE
    original_day = sim_module.DAY_OF_WEEK_PROFILE
    original_hour = sim_module.HOUR_OF_DAY_PROFILE
    try:
        sim_module.TRIAGE = (("single", 1.0, mean_service, 0),)
        sim_module.DAY_OF_WEEK_PROFILE = tuple([1.0] * 7)
        sim_module.HOUR_OF_DAY_PROFILE = tuple([1.0] * 24)
        return simulate(
            flat_roster(servers), weeks=weeks,
            arrivals_per_hour_mean=arrival_rate, seed=seed,
            balk_after_hours=10_000.0,   # no balking in the analytic case
        )
    finally:
        sim_module.TRIAGE = original_triage
        sim_module.DAY_OF_WEEK_PROFILE = original_day
        sim_module.HOUR_OF_DAY_PROFILE = original_hour


class TestValidationAgainstClosedForm(unittest.TestCase):
    """The check that has to pass before any other number means anything."""

    CASES = [
        # (servers, arrivals/hr, mean service hrs) -> utilisation
        (4, 3.0, 1.0),    # rho = 0.75
        (6, 4.0, 1.2),    # rho = 0.80
        (8, 5.0, 1.3),    # rho ~ 0.81
    ]

    def test_mean_wait_matches_analytic_within_tolerance(self):
        for servers, lam, service in self.CASES:
            with self.subTest(servers=servers):
                analytic = MMc(lam, 1 / service, servers)
                self.assertLess(analytic.utilisation, 1.0)
                result = mmc_sim(servers, lam, service, weeks=40, seed=11)

                expected = analytic.mean_wait
                actual = result.mean_wait
                rel = abs(actual - expected) / max(expected, 1e-9)
                self.assertLess(
                    rel, 0.15,
                    f"c={servers}: analytic Wq={expected:.4f}h, "
                    f"simulated={actual:.4f}h ({rel:.1%} off)",
                )

    def test_the_TAIL_also_matches_analytic(self):
        # Matching the mean is easy and insufficient: the project's entire
        # claim is about the tail, so the tail is what must be validated.
        servers, lam, service = 6, 4.0, 1.2
        analytic = MMc(lam, 1 / service, servers)
        result = mmc_sim(servers, lam, service, weeks=60, seed=5)

        for p in (50, 90, 95):
            expected = analytic.wait_percentile(p)
            actual = result.percentile(p)
            self.assertLess(
                abs(actual - expected), max(0.12, expected * 0.25),
                f"p{p}: analytic {expected:.3f}h, simulated {actual:.3f}h",
            )

    def test_erlang_c_is_correct_at_known_points(self):
        # A saturated system always waits.
        self.assertAlmostEqual(erlang_c(2, 2.0), 1.0)
        self.assertAlmostEqual(erlang_c(1, 0.5), 0.5, places=6)
        self.assertGreater(erlang_c(3, 2.5), erlang_c(4, 2.5),
                           "more servers must reduce the probability of waiting")

    def test_THE_PREMISE_the_tail_is_far_worse_than_the_mean(self):
        analytic = MMc(4.0, 1 / 1.2, 6)
        ratio = analytic.wait_percentile(95) / analytic.mean_wait
        self.assertGreater(
            ratio, 3.0,
            "if p95 were close to the mean, staffing to the average would be "
            "fine and this project would have no point",
        )


class TestSeasonality(unittest.TestCase):
    def test_arrival_intensity_varies_by_hour_and_day(self):
        weekday_3am = intensity(3)
        saturday_evening = intensity(5 * 24 + 19)
        self.assertGreater(saturday_evening, weekday_3am * 3)

    def test_the_profile_covers_the_whole_week(self):
        values = [intensity(h) for h in range(HOURS_PER_WEEK)]
        self.assertEqual(len(values), 168)
        self.assertTrue(all(v > 0 for v in values))

    def test_simulated_waits_are_worse_in_peak_hours(self):
        result = simulate(flat_roster(9), weeks=10, arrivals_per_hour_mean=7.0,
                          seed=3)
        peak_hours = [h for h in range(HOURS_PER_WEEK) if intensity(h) > 1.3]
        quiet_hours = [h for h in range(HOURS_PER_WEEK) if intensity(h) < 0.6]

        peak = [w for h in peak_hours for w in result.waits_by_hour_of_week.get(h, [])]
        quiet = [w for h in quiet_hours
                 for w in result.waits_by_hour_of_week.get(h, [])]
        self.assertGreater(len(peak), 50)
        self.assertGreater(len(quiet), 20)
        self.assertGreater(statistics.mean(peak), statistics.mean(quiet) * 1.5,
                           "a flat roster should visibly struggle at peak")


class TestRosterComparison(unittest.TestCase):
    def test_THE_HEADLINE_reshaping_a_roster_at_ZERO_cost_cuts_the_tail(self):
        flat = flat_roster(9)
        shaped = demand_shaped_roster(flat.total_hours)

        self.assertEqual(flat.total_hours, shaped.total_hours,
                         "the comparison is only interesting if it is free")

        a = simulate(flat, weeks=12, arrivals_per_hour_mean=7.0, seed=17)
        b = simulate(shaped, weeks=12, arrivals_per_hour_mean=7.0, seed=17)

        self.assertLess(b.percentile(95), a.percentile(95),
                        "demand-shaped staffing should cut the p95 wait")
        self.assertLess(b.breach_rate, a.breach_rate)

    def test_the_MEAN_understates_the_improvement_IN_HOURS(self):
        """Report the saving in hours, not as a percentage.

        The percentage change can be similar for the mean and the tail, and
        quoting percentages makes the two look equivalent. They are not: the
        tail falls by a far larger number of ACTUAL HOURS, and hours waited is
        what the four-hour target, the ambulance handover and the patient all
        experience.
        """
        flat = flat_roster(9)
        shaped = demand_shaped_roster(flat.total_hours)
        a = simulate(flat, weeks=12, arrivals_per_hour_mean=7.0, seed=17)
        b = simulate(shaped, weeks=12, arrivals_per_hour_mean=7.0, seed=17)

        mean_saved = a.mean_wait - b.mean_wait
        p95_saved = a.percentile(95) - b.percentile(95)
        self.assertGreater(
            p95_saved, mean_saved * 1.5,
            f"p95 saved {p95_saved:.2f}h vs mean {mean_saved:.2f}h - the tail "
            f"is where the time actually is",
        )

    def test_a_roster_must_cover_the_whole_week(self):
        with self.assertRaises(ValueError):
            Roster("bad", tuple([5] * 24))

    def test_moving_staff_is_cost_neutral(self):
        base = flat_roster(9)
        moved = base.moved(from_hour=2 * 24 + 3, to_hour=5 * 24 + 20, count=2)
        self.assertEqual(base.total_hours, moved.total_hours)


class TestOutputs(unittest.TestCase):
    def setUp(self):
        self.result = simulate(flat_roster(9), weeks=8,
                               arrivals_per_hour_mean=7.0, seed=23)

    def test_percentiles_are_ordered(self):
        r = self.result
        self.assertLessEqual(r.percentile(50), r.percentile(90))
        self.assertLessEqual(r.percentile(90), r.percentile(95))
        self.assertLessEqual(r.percentile(95), r.percentile(99))

    def test_breach_rate_is_consistent_with_the_waits(self):
        r = self.result
        manual = sum(1 for w in r.waits if w > BREACH_HOURS) / len(r.waits)
        self.assertAlmostEqual(r.breach_rate, manual, places=9)

    def test_higher_acuity_patients_are_seen_sooner(self):
        by_cat = self.result.waits_by_category
        if "resus" in by_cat and "non_urgent" in by_cat:
            self.assertLess(statistics.mean(by_cat["resus"]),
                            statistics.mean(by_cat["non_urgent"]),
                            "triage priority must actually be honoured")

    def test_the_simulation_is_reproducible(self):
        a = simulate(flat_roster(9), weeks=4, arrivals_per_hour_mean=7.0, seed=99)
        b = simulate(flat_roster(9), weeks=4, arrivals_per_hour_mean=7.0, seed=99)
        self.assertEqual(a.waits, b.waits,
                         "a non-reproducible simulation is not a bug report")

    def test_a_surge_costs_more_HOURS_in_the_tail_than_at_the_mean(self):
        base = simulate(flat_roster(9), weeks=8, arrivals_per_hour_mean=7.0, seed=7)
        surge = simulate(flat_roster(9), weeks=8, arrivals_per_hour_mean=7.0,
                         seed=7, surge_multiplier=1.25)
        mean_added = surge.mean_wait - base.mean_wait
        p95_added = surge.percentile(95) - base.percentile(95)
        self.assertGreater(p95_added, 0.0)
        self.assertGreater(
            p95_added, mean_added,
            f"a 25% surge added {mean_added:.2f}h to the mean but "
            f"{p95_added:.2f}h to p95 - non-linearity is the whole point",
        )
        self.assertGreater(surge.breach_rate, base.breach_rate)


if __name__ == "__main__":
    unittest.main()

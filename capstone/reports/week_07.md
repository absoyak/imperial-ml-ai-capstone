# Week 7 — Boundary Expansion and Local Peak Refinement

## Overview

Week 7 focused on consolidating the strongest regions discovered so far
while carefully extending the search around promising peaks.

Following the strong improvements of Week 6, particularly in Functions 5 and 4,
the optimisation strategy shifted toward deeper local refinement in confirmed
high-performing regions, while maintaining selective exploration in functions
that remained unstable.

Key adjustments included:
- Continued Expected Improvement (EI) exploitation for strong regions (F5, F7)
- Slightly expanded local search around the current best candidates
- Variance-driven exploration for poorly performing landscapes (notably F6)
- Moderate UCB exploration in functions with partially recovered behaviour (F2, F8)

The objective this week was to expand confirmed peak regions further while
testing whether controlled exploration could uncover additional improvements.

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Change vs Week 6 |
|----------|----------------|--------|------------------|------------------|
| F1 | `[0.888172, 0.768020]` | `~0` | EI | No meaningful change |
| F2 | `[0.708064, 0.895827]` | `0.5228` | UCB | Slight improvement |
| F3 | `[0.766569, 0.823794, 0.002273]` | `-0.1198` | EI | Slight decrease |
| F4 | `[0.445779, 0.391676, 0.386881, 0.456083]` | `-0.7241` | UCB | Decrease |
| F5 | `[0.042789, 0.999999, 0.999999, 0.999999]` | `4440.54` | EI | Major improvement |
| F6 | `[0.977713, 0.021016, 0.276503, 0.871049, 0.993834]` | `-2.2940` | Variance | Decrease |
| F7 | `[0.141990, 0.235001, 0.291233, 0.279310, 0.304045, 0.653243]` | `2.96` | EI | Strong improvement |
| F8 | `[0.119430, 0.114060, 0.183865, 0.027102, 0.390019, 0.787933, 0.469002, 0.878341]` | `9.65` | UCB | Improvement |

---

## Summary

Week 7 produced several strong improvements while also revealing the limits
of the current surrogate model in certain landscapes.

The most significant progress again occurred in **Function 5**, which increased
from 3555.59 to **4440.54**. This confirms that the optimiser continues to move
along a strong high-value ridge near the boundary of the search space. The
observed behaviour strongly suggests that the optimum lies close to the upper
limits of several variables, and the EI-driven local refinement is effectively
expanding that peak region.

**Function 7** also showed a clear improvement, rising from 2.73 to **2.96**.
This reinforces the view that this landscape is relatively smooth and well
captured by the Gaussian Process surrogate, allowing consistent incremental
refinement.

**Function 8** improved slightly from 9.45 to **9.65**, suggesting that the
UCB-based strategy continues to operate near a productive region of the
search space.

**Function 2** showed a modest increase from 0.5071 to **0.5228**, indicating
slow recovery after the instability observed in earlier weeks. While still
below the initial peak discovered in Week 1, the improvement suggests that
uncertainty-aware exploration remains beneficial in this landscape.

In contrast, **Function 4** regressed this week after the major improvement
observed in Week 6. This suggests that the Week 6 point may lie near a narrow
high-value region, and further exploration around it did not immediately
identify similarly strong areas.

**Function 6** also decreased significantly despite the use of variance-based
exploration. This indicates that the function landscape may contain large
low-value regions or that the surrogate model remains poorly calibrated for
this function.

**Function 3** remained relatively unstable and again produced a slightly
worse result compared to the previous week.

Finally, **Function 1** continued to produce near-zero outputs, reinforcing
the hypothesis that this landscape either contains extremely narrow peaks
or is largely flat within the explored regions.

Overall, Week 7 further strengthens the conclusion that the optimisation
framework performs extremely well in structured landscapes with strong
peaks (notably F5 and F7). At the same time, it highlights the continuing
difficulty of modelling landscapes that are either highly irregular or
contain very narrow optimal regions.

Future iterations will likely focus on:
- Continued peak expansion for F5
- Stable refinement in the F7 region
- Careful recovery attempts for F2
- Re-evaluating exploration strategies for F3 and F6
- Testing whether the high-value region in F4 can be rediscovered
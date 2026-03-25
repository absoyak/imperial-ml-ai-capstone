# Week 8 — Peak Consolidation and Selective Recovery

## Overview

Week 8 focused on consolidating peak regions identified in earlier weeks
while testing recovery strategies for functions that had shown instability.

Following the mixed results of Week 7, the strategy shifted further toward
tight local exploitation in confirmed high-value regions, with controlled
UCB-based exploration reserved for functions where EI had stalled.

Key adjustments included:
- Continued EI exploitation for F3, F5 and F7 near confirmed peaks
- UCB-based recovery for F4 and F8 with reduced kappa
- Renewed uncertainty-driven search for F6
- Maintained single-centre constraint for F2

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Change vs Week 7 |
|----------|----------------|--------|------------------|------------------|
| F1 | `[0.335887, 0.996204]` | `~0` | Spread | No meaningful change |
| F2 | `[0.999712, 0.417581]` | `0.0222` | UCB | Significant decrease |
| F3 | `[0.279800, 0.302235, 0.482819]` | `-0.0168` | EI | New best |
| F4 | `[0.433573, 0.409382, 0.384042, 0.353678]` | `0.4821` | UCB | New best |
| F5 | `[0.076313, 0.999999, 0.999999, 0.999999]` | `4440.77` | EI | Marginal improvement |
| F6 | `[0.609913, 0.272532, 0.650951, 0.875100, 0.000000]` | `-0.6702` | UCB | Slight decrease |
| F7 | `[0.127235, 0.182925, 0.237518, 0.309570, 0.300939, 0.650319]` | `2.8798` | EI | Slight decrease |
| F8 | `[0.061569, 0.088661, 0.146334, 0.109563, 0.378843, 0.769222, 0.571593, 0.921671]` | `9.5432` | UCB | Decrease |

---

## Summary

Week 8 delivered meaningful progress in two key functions while exposing
ongoing instability in others.

**Function 3** achieved a new best of **-0.0168**, its strongest result to date.
This confirms that the EI-driven search is gradually refining the surrogate
model toward a productive region in this landscape, even if progress has
been slow and non-monotonic across weeks.

**Function 4** also set a new best at **0.4821**, recovering the positive region
first identified in Week 6 and extending it further. This is encouraging given
the regression observed in Week 7, and suggests the landscape around this
region is more accessible than previously assumed.

**Function 5** produced **4440.77**, marginally exceeding its Week 7 output.
The function continues to operate near the upper boundary of the search space,
and the surrogate model appears well calibrated for this region. With diminishing
incremental gains, the trajectory suggests proximity to the true maximum.

**Function 7** regressed slightly from 2.96 to **2.88**. While this is below the
Week 7 peak, the values remain within a consistently strong range, suggesting
that the surrogate model is operating near a broad, stable peak rather than
a narrow spike.

**Function 8** also decreased from 9.65 to **9.54**. The UCB strategy explored
a different region of the 8-dimensional space this week, which did not outperform
the current best. With five weeks remaining, tighter local refinement around
the confirmed best region is warranted.

**Function 2** regressed significantly to 0.022. The single-centre UCB strategy
directed the query away from the established high-value region near
(0.71, 0.93), indicating that UCB with a higher kappa introduced too much
exploration even under localised sampling. Reducing kappa further is the
logical next step.

**Function 6** remained below its best despite renewed UCB exploration.
This landscape continues to resist stable improvement, and future queries
will prioritise EI over UCB to reduce the risk of large excursions.

**Function 1** continued to produce near-zero outputs. Spread-based sampling
has systematically covered previously unexplored regions without discovering
any meaningful signal.

Overall, Week 8 reinforced the view that exploitation is now more valuable
than exploration across most functions. With five weeks remaining, future
strategy will prioritise tight local refinement in F3, F4, F5 and F7,
while making conservative recovery attempts in F2 and F6.

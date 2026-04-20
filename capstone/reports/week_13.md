# Week 13 — Campaign Conclusion and Final Records

## Overview

Week 13 marked the final round of the 13-week Black-Box Optimisation campaign.

With no further queries remaining after this round, every configuration
decision was committed to maximum exploitation around confirmed best regions,
with one deliberate exception: Function 1 received a final exploratory attempt
at a completely unexplored region of the search space.

Key adjustments included:
- Reduced `filterTopK` limits for F2 and F6 to 10 observations each for tighter
  local GP calibration
- Added `filterTopK(10)` for F7 to exclude early exploratory queries that were
  misleading the surrogate model
- F4 switched from UCB to EI with single-centre constraint around the Week 8 peak
- F1 `buildCandidatesF1` extended with a third centre at the mirror point
  (0.580, 0.536) — the reflection of the Week 6 signal across the unit square
- F5 continued with the fixed boundary submission that produced the Week 12 record
- All other functions used the tightest local standard deviations of the campaign

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Change vs Week 12 |
|----------|----------------|--------|------------------|-------------------|
| F1 | `[0.819958, 0.724876]` | `~0` | EI | No meaningful change |
| F2 | `[0.706765, 0.934636]` | `0.4383` | EI | Decrease |
| F3 | `[0.443494, 0.658545, 0.504177]` | `-0.0118` | EI | Slight decrease |
| F4 | `[0.443316, 0.406619, 0.349807, 0.361772]` | `0.3402` | EI | Decrease |
| F5 | `[0.999999, 0.999999, 0.999999, 0.999999]` | `8662.40` | Fixed | Stable — best maintained |
| F6 | `[0.625630, 0.458317, 0.633340, 0.713119, 0.000000]` | `-0.4602` | UCB | Slight decrease |
| F7 | `[0.152286, 0.148893, 0.378958, 0.300823, 0.304431, 0.682178]` | `3.1810` | EI | **New best** |
| F8 | `[0.127473, 0.101396, 0.098512, 0.088212, 0.387150, 0.797850, 0.472005, 0.869865]` | `9.6601` | UCB | Stable |

---

## Summary

Week 13 delivered one new record and confirmed the stability of the strongest
regions identified in previous weeks.

**Function 7** set a new best at **3.1810**, its highest value across the
entire 13-week campaign. The addition of `filterTopK(10)` restricted the GP
fit to the most recent high-value observations and eliminated the influence
of early exploratory queries that had been pulling the surrogate toward
lower-value predictions. The resulting suggestion landed in the confirmed
high-value region and produced a cleaner refinement than any previous week.

**Function 5** returned **8662.40**, maintaining the record set in Week 12.
The fixed boundary submission at (0.999999, 0.999999, 0.999999, 0.999999)
locked in the strongest single-function result of the campaign and confirmed
that the true optimum lies at or very near the corner of the search space.

**Function 2** returned **0.4383**, below its best of 0.663. The surrogate
landed within the established high-value region but did not hit the precise
peak. With the final query spent, the Week 10 record remains the campaign best.

**Function 3** returned **-0.0118**, slightly below its Week 12 best of -0.0088.
The EI suggestion remained in the correct region but did not produce further
improvement. The Week 12 record stands as the final best.

**Function 4** returned **0.3402**, below its Week 8 best of 0.482. The switch
to EI with a single-centre constraint produced a query closer to the Week 8
peak than recent UCB attempts, but did not reproduce the exact value. The
surrogate model never achieved reliable calibration for this landscape
across all 13 weeks.

**Function 6** returned **-0.4602**, slightly below its Week 12 best of -0.413.
The tighter UCB configuration sampled a nearby region but did not improve
on the previous round. The Week 12 record stands as the final best.

**Function 8** returned **9.6601**, effectively stable relative to the Week 9
best of 9.675. The 8-dimensional landscape limited how precisely the GP could
refine around the peak, but the UCB strategy consistently targeted the
correct region across the final four weeks.

**Function 1** produced **~0** for the thirteenth consecutive week. The
dual-centre sampling with a speculative mirror point at (0.580, 0.536) did
not reveal any new signal. Across 13 rounds, this function produced no
identifiable high-value region.

---

## Final Campaign Summary

| Function | Dim | Week 1 Best | Final Best | Overall Change |
|----------|-----|-------------|------------|----------------|
| F1 | 2 | ~0 | ~0 | No improvement |
| F2 | 2 | 0.641 | 0.663 | +3.4% |
| F3 | 3 | -0.483 | -0.0088 | Major improvement |
| F4 | 4 | -31.18 | +0.482 | Major improvement |
| F5 | 4 | 1163.7 | 8662.4 | +644% |
| F6 | 5 | -2.75 | -0.413 | Major improvement |
| F7 | 6 | 2.27 | 3.181 | +40.1% |
| F8 | 8 | 9.31 | 9.675 | +3.9% |

Seven out of eight functions showed meaningful improvement over the campaign.
The strongest result was Function 5, where the boundary exploitation strategy
delivered a 644% improvement over the Week 1 baseline. Functions 3, 4 and 6
recovered from strongly negative initial values to produce substantially
better outcomes. Functions 2 and 8 achieved incremental improvements that
confirmed the robustness of the exploitation strategy for smooth, well-modelled
landscapes. Function 7 delivered steady gains throughout the campaign and
produced a new record in the final round.

Function 1 remained the single unresolved case — 13 rounds of varied
acquisition strategies (variance, spread, EI) and diverse query points
across the 2D space produced no identifiable high-value region, suggesting
either a flat landscape within the accessible domain or a peak so narrow
that it was never sampled.

---

## Reflection on the Campaign

The optimisation strategy evolved from uniform exploration in Week 1 to
adaptive, function-specific exploitation by the final weeks. The most
important lesson was that no single acquisition rule works across all
functions — each landscape required its own configuration, and those
configurations had to be informed by accumulated observations rather than
applied uniformly.

The most significant strategic decisions in retrospect were:
- Introducing y-normalisation in Week 5 to stabilise GP inference
- Switching F5 to boundary-pinned candidates after recognising the ridge structure
- Introducing `filterTopK` in Week 12 to reduce the influence of outdated observations
- Accepting that F1 and F4 were not reliably modellable with the available query budget

The Week 12 fixed submission for F5 and the Week 13 `filterTopK` expansion
for F7 were the two late-campaign changes that delivered the largest final
improvements, demonstrating that strategic adjustments remained valuable
even in the final rounds.

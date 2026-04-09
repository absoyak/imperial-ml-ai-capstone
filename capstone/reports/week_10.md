# Week 10 — Final Exploitation Phase and New Records

## Overview

Week 10 marked the beginning of the final three-week exploitation phase,
with the strategy now fully committed to tight local refinement across all functions.

Following the mixed results of Week 9, several key changes were introduced:
- Function 1 switched from spread-based sampling to EI targeting the only
  observed non-zero region
- Function 2 moved from UCB to EI with localStd reduced to 0.006
- Function 5 introduced a boundary-pinned candidate generation strategy,
  keeping x2, x3 and x4 fixed at 0.999999 while allowing x1 to vary freely
- localStd tightened further for F3, F7 and F8
- Function 4 switched from UCB to EI with a single-centre constraint

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Change vs Week 9 |
|----------|----------------|--------|------------------|------------------|
| F1 | `[0.231431, 0.825513]` | `~0` | EI | No meaningful change |
| F2 | `[0.704578, 0.936115]` | `0.6632` | EI | **New best** |
| F3 | `[0.576636, 0.567316, 0.581224]` | `-0.0376` | EI | Decrease |
| F4 | `[0.417305, 0.485612, 0.233663, 0.449842]` | `-3.0524` | EI | Significant decrease |
| F5 | `[0.059579, 0.999999, 0.999999, 0.999999]` | `4440.62` | EI | Stable |
| F6 | `[0.696702, 0.195279, 0.573176, 0.777706, 0.000000]` | `-0.5885` | EI | Slight decrease |
| F7 | `[0.154664, 0.176139, 0.328620, 0.309794, 0.301649, 0.687208]` | `3.0801` | EI | **New best** |
| F8 | `[0.098251, 0.122265, 0.137629, 0.072555, 0.342077, 0.838028, 0.479244, 0.897349]` | `9.6091` | EI | Slight decrease |

---

## Summary

Week 10 produced two significant new records while also exposing the limits
of the surrogate model in certain landscapes.

**Function 2** achieved a new best of **0.6632**, surpassing the original Week 1
peak of 0.641 for the first time in the campaign. The switch from UCB to EI,
combined with a very tight localStd of 0.006 and single-centre sampling around
the established high-value region near (0.71, 0.93), produced the most reliable
result this function has delivered since Week 1. This confirms that conservative
exploitation with a well-calibrated surrogate is the correct strategy for
this landscape at this stage.

**Function 7** set a new best at **3.0801**, its strongest result across the
entire campaign. The EI-driven local refinement with tightened localStd
continues to produce consistent incremental improvements, confirming that
this landscape is relatively smooth and well captured by the Gaussian Process
surrogate.

**Function 5** produced **4440.62**, remaining stable near its peak. The
boundary-pinned candidate strategy — x2, x3 and x4 fixed at 0.999999 —
successfully avoided the regression seen in Week 9 when x3 drifted to 0.953.
This confirms that the function has a narrow ridge near the upper boundary
rather than a broad plateau, and the strategy will be maintained in the
remaining weeks.

**Function 8** regressed slightly to **9.6091**, below its Week 9 best.
The EI suggestion moved to a different region of the 8-dimensional space
that did not outperform the current best. Tightening local sampling further
around the Week 9 best point is warranted for the remaining weeks.

**Function 3** decreased from -0.0106 to **-0.0376**. The EI suggestion
moved away from the Week 9 best region, indicating that the surrogate model
may not be reliably calibrated around the recent best. Returning to the
Week 9 centre with a tighter localStd is the priority for Week 11.

**Function 4** regressed significantly to **-3.0524**. Despite switching to
EI with a single-centre constraint, the surrogate model remains poorly
calibrated in this landscape with sigma values consistently above 5.0.
No reliable exploitation strategy has emerged across ten weeks, and this
function is unlikely to yield further improvement in the remaining rounds.

**Function 6** showed a marginal decrease to **-0.5885**. The landscape
continues to resist stable improvement, and the function remains close to
its cumulative best of -0.476 without making further progress.

**Function 1** again produced a near-zero output. The switch from spread
to EI did not yield a meaningful signal. With three weeks remaining, the
probability of discovering a productive region in this landscape is low.

Overall, Week 10 confirmed the strength of the exploitation strategy for
Functions 2, 5 and 7, while reinforcing the difficulty of making progress
in Functions 1, 4 and 6. With three weeks remaining, the priority is to
consolidate and extend the records in F2 and F7, maintain F5 at its boundary
peak, and recover F3 and F8 to their best observed values.

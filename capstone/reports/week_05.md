# Week 5 — Aggressive Local Refinement and Peak Expansion

## Overview

Week 5 focused on aggressive local refinement in confirmed high-performing regions,
while attempting recovery in unstable functions.

Following the strong improvements of Week 4, the strategy shifted further toward
controlled exploitation in functions that demonstrated consistent upward trends,
particularly in higher dimensions.

Key adjustments included:
- Extremely concentrated local sampling for high-performing functions
- Expected Improvement (EI) as the primary refinement mechanism
- Limited exploration for functions with weak or flat signals
- Continued function-specific kernel and acquisition tuning

The primary objective was to maximise gains in confirmed promising regions
while testing whether unstable functions (notably F2 and F6) could recover.

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Change vs Week 4 |
|----------|----------------|--------|------------------|------------------|
| F1 | `[0.673459, 0.000722]` | `~0` | Variance | No meaningful change |
| F2 | `[0.780704, 0.944251]` | `0.0718` | EI | Decrease |
| F3 | `[0.351520, 0.176589, 0.456402]` | `-0.0312` | EI | Slight decrease |
| F4 | `[0.414516, 0.372734, 0.513186, 0.304630]` | `-2.08` | UCB | Stable |
| F5 | `[0.000000, 0.882742, 0.944023, 0.999999]` | `2603.66` | EI | Major improvement |
| F6 | `[0.698217, 0.271226, 0.498420, 0.760055, 0.138323]` | `-0.6384` | EI | Decrease |
| F7 | `[0.065907, 0.268693, 0.303928, 0.254584, 0.306775, 0.699677]` | `2.69` | EI | Strong improvement |
| F8 | `[0.123737, 0.130854, 0.157668, 0.166516, 0.400111, 0.799520, 0.497221, 0.942502]` | `9.64` | UCB | Slight improvement |

---

## Summary

Week 5 delivered the largest single-function improvement so far.

Function 5 increased dramatically from 2100.97 to 2603.66,
confirming that the aggressive local exploitation strategy is highly effective
in stable high-value landscapes. The optimiser appears to be expanding the peak
rather than merely hovering near it.

Function 7 also showed strong upward movement (2.41 → 2.69),
indicating continued refinement in a well-calibrated region.

Function 8 achieved a modest but consistent increase, suggesting
that UCB-based exploration remains productive in that landscape.

Function 4 stabilised near its improved Week 4 level,
while Function 1 continues to show no meaningful signal.

However, Functions 2 and 6 remain problematic.

Function 2 continues to drift away from its Week 1 peak,
despite attempts at concentrated exploitation. This reinforces the hypothesis that:
- The optimum region may be extremely narrow, or
- The surrogate model is miscalibrated in that landscape.

Function 6 also regressed, indicating that over-exploitation may be occurring
in a region that is not globally optimal.

Overall, Week 5 confirms that the optimisation framework performs strongly
in structured and higher-dimensional landscapes (F5, F7, F8),
but still struggles in landscapes that are either highly narrow-peaked
or poorly approximated by the current Gaussian Process configuration.

The optimisation process is now clearly exploit-driven in strong regions.
Future iterations will likely require:
- Continued aggressive refinement for F5 and F7
- A revised recovery strategy for F2
- More cautious handling of F6 to avoid premature local convergence
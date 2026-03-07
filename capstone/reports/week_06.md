# Week 6 — Peak Expansion and Strategic Recovery

## Overview

Week 6 focused on preserving the strongest exploitation regions while introducing
targeted recovery strategies for unstable functions.

Following the aggressive local refinement of Week 5, the strategy diverged further
per function. High-performing landscapes continued to receive concentrated
Expected Improvement (EI) exploitation, while unstable or poorly calibrated
functions were handled with more cautious uncertainty-driven approaches.

Key adjustments included:
- Continued aggressive exploitation in confirmed peak regions (F5, F7)
- UCB-based recovery attempts for unstable functions (notably F2 and F4)
- Spread-based sampling for flat or low-signal landscapes (F1)
- Slight reduction of over-exploitation pressure in mid-performing functions

The primary objective was to expand the strongest peaks while testing whether
uncertainty-aware exploration could recover performance in unstable functions.

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Change vs Week 5 |
|----------|----------------|--------|------------------|------------------|
| F1 | `[0.420436, 0.463773]` | `-0.0069` | Spread | No meaningful change |
| F2 | `[0.687950, 0.985472]` | `0.5071` | UCB | Partial recovery |
| F3 | `[0.445626, 0.767936, 0.000000]` | `-0.1111` | EI | Decrease |
| F4 | `[0.387694, 0.371025, 0.341002, 0.355560]` | `0.4079` | UCB | Major improvement |
| F5 | `[0.000000, 0.930257, 0.995528, 0.999999]` | `3555.59` | EI | Major improvement |
| F6 | `[0.820687, 0.302272, 0.592412, 0.690131, 0.039193]` | `-0.7006` | UCB | Decrease |
| F7 | `[0.122659, 0.230033, 0.286637, 0.216736, 0.337719, 0.702179]` | `2.73` | EI | Slight improvement |
| F8 | `[0.147507, 0.091806, 0.149744, 0.120066, 0.376664, 0.899270, 0.581779, 0.812084]` | `9.45` | UCB | Slight decrease |

---

## Summary

Week 6 delivered the strongest improvement observed so far in the optimisation process.

Function 5 achieved a dramatic increase from 2603.66 to 3555.59.  
This confirms that the optimisation framework is successfully expanding the
high-value region previously identified. The behaviour suggests that the optimum
may lie close to the boundary of the search space, and the concentrated EI-based
local refinement continues to move along that ridge.

Function 4 also showed a major improvement, jumping from approximately -2.08
to 0.41. This indicates that the UCB-based recovery strategy successfully escaped
a poor local region and discovered a significantly better part of the landscape.

Function 7 continued its steady upward trend (2.69 → 2.73), reinforcing the
hypothesis that this landscape is stable and well approximated by the current
surrogate configuration.

Function 2 partially recovered compared to Week 5, increasing from 0.0718
to 0.5071. While still below its Week 1 peak, the result suggests that
uncertainty-aware exploration is more effective than aggressive exploitation
in this landscape.

Function 1 again produced a near-zero output, supporting the assumption that
this landscape is largely flat or poorly structured. Spread-based sampling
remains the most reasonable strategy under such conditions.

However, Functions 3 and 6 regressed this week, indicating that the surrogate
model may still be over-exploiting weak local signals in those landscapes.

Function 8 slightly decreased from 9.64 to 9.45, but the result remains close
to the previously discovered peak, suggesting that the optimiser is still
operating near a strong region.

Overall, Week 6 confirms that the optimisation framework is highly effective
in structured landscapes with clear peaks (notably F5 and F7), while recovery
strategies using uncertainty-aware sampling can also unlock significant
improvements in previously unstable functions such as F4.

The optimisation process is now clearly dominated by peak expansion in strong
regions combined with targeted recovery attempts elsewhere.

Future iterations will likely focus on:
- Continued peak expansion for F5
- Stable refinement around F7
- Further recovery attempts in F2
- Reconsideration of exploration strategies for F3 and F6
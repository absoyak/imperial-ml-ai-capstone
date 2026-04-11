# Week 11 — Final Exploitation and Stability Limits

## Overview

Week 11 continued the final exploitation phase, with the strategy now almost entirely focused on tight local refinement around previously discovered high-value regions.

With only one iteration remaining, the approach prioritised:
- Maximum exploitation of confirmed peak regions (F5, F7, F8)
- Maintaining stability in functions that recently achieved strong results (F2, F3)
- Minimal exploration, except where recovery still appeared possible

Key adjustments included:
- Full EI-based optimisation across all functions
- Extremely tight local sampling around best-known points
- Continued boundary-focused search for F5
- Removal of most exploratory behaviour to avoid unnecessary drift

The primary objective was to consolidate gains and push peak values slightly further where possible.

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Change vs Week 10 |
|----------|----------------|--------|------------------|------------------|
| F1 | `[0.422118, 0.920589]` | `~0` | EI | No meaningful change |
| F2 | `[0.703389, 0.933442]` | `0.4859` | EI | Decrease |
| F3 | `[0.994652, 0.584591, 0.639685]` | `-0.0796` | EI | Decrease |
| F4 | `[0.322128, 0.440024, 0.294843, 0.304790]` | `-2.1659` | EI | Decrease |
| F5 | `[0.311343, 0.999999, 0.999999, 0.999999]` | `4478.77` | EI | New best |
| F6 | `[0.641520, 0.239868, 0.668150, 0.765107, 0.165406]` | `-0.5105` | EI | Slight improvement |
| F7 | `[0.185780, 0.190846, 0.321921, 0.359268, 0.324116, 0.702584]` | `2.9052` | EI | Slight decrease |
| F8 | `[0.083409, 0.108674, 0.131521, 0.098181, 0.393266, 0.813641, 0.475379, 0.834536]` | `9.6568` | EI | Near best |

---

## Summary

Week 11 confirmed that the optimisation process has largely converged in several functions, while also highlighting the limits of further improvement in others.

**Function 5** achieved a new best of **4478.77**, slightly exceeding its previous peak. The boundary-constrained optimisation strategy continues to perform exceptionally well, reinforcing the conclusion that the global optimum lies along a narrow ridge at the upper boundary of the search space. Gains are now marginal, indicating proximity to the true maximum.

**Function 8** produced **9.6568**, remaining very close to its best observed value. The optimiser is clearly operating within a stable high-value region, and further improvements are likely to be incremental.

**Function 7** returned **2.9052**, slightly below its peak but still within a consistently strong range. This behaviour suggests that the function contains a relatively smooth plateau rather than a sharp optimum.

**Function 2** decreased to **0.4859** after achieving a new best in Week 10. The tight EI-driven exploitation may have overcommitted to a local region, reinforcing the idea that this landscape contains a narrow optimum that is sensitive to small deviations.

**Function 6** showed a slight improvement to **-0.5105**, but remains close to its long-term plateau. Despite multiple strategy changes across weeks, no consistent upward trend has been established.

**Function 3** regressed significantly from its best, suggesting that the surrogate model is not reliably capturing the local structure around the optimum. The jump to a distant region indicates overconfidence in poorly supported areas of the search space.

**Function 4** again produced a strongly negative result, confirming that the Gaussian Process surrogate has struggled to model this landscape effectively throughout the campaign. Previous positive results appear to lie in narrow regions that are difficult to rediscover.

**Function 1** continued to produce near-zero outputs, confirming the lack of a usable signal within the explored domain.

Overall, Week 11 demonstrates that the optimisation has effectively converged in structured landscapes (F5, F7, F8), while remaining limited by surrogate uncertainty and landscape complexity in others (F3, F4, F6).

With one week remaining, the final iteration will focus on:
- Maximising the boundary peak in F5
- Attempting minor refinements in F7 and F8
- Stabilising F2 near its Week 10 optimum
- Avoiding unnecessary exploration in unstable functions
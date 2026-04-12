# Week 12 — Final Round: Structural Improvements and Campaign Conclusion

## Overview

Week 12 marked the final round of the Black-Box Optimisation campaign.

With the query budget fully exhausted, the strategy committed entirely to
maximum exploitation across all functions. Several structural improvements
were introduced to the codebase to extract the most value from the remaining
observations:

- Candidate count increased to 200,000 for the finest search resolution to date
- `getModelParams()` introduced to manage per-function kernel length scales
  and noise levels as explicit configuration rather than inline conditionals
- `filterTopK()` introduced to fit the GP only on the most recent high-value
  observations for selected functions, reducing the influence of early
  exploratory queries that no longer reflected the current search region
- `buildCandidatesF1()` introduced — dual-centre sampling around both the
  best-observed point and the most informative negative signal
- Function 5 fixed to `0.999999-0.999999-0.999999-0.999999` based on
  accumulated boundary ridge evidence across the final four weeks
- Functions 2 and 3 used single-centre EI with the tightest local standard
  deviations of the campaign (0.005 and 0.004 respectively)
- Function 8 switched from EI to UCB with kappa=0.0 and wider local sampling
  after EI repeatedly failed to calibrate in 8-dimensional space

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Change vs Week 11 |
|----------|----------------|--------|------------------|-------------------|
| F1 | `[0.171556, 0.650572]` | `~0` | EI | No meaningful change |
| F2 | `[0.701055, 0.938577]` | `0.6150` | EI | Slight decrease |
| F3 | `[0.429563, 0.670377, 0.509394]` | `-0.0088` | EI | **New best** |
| F4 | `[0.386773, 0.383302, 0.386447, 0.359155]` | `0.1929` | UCB | Decrease |
| F5 | `[0.999999, 0.999999, 0.999999, 0.999999]` | `8662.40` | Fixed | **New best — major jump** |
| F6 | `[0.597844, 0.416470, 0.723257, 0.661940, 0.121390]` | `-0.4130` | UCB | **New best** |
| F7 | `[0.146038, 0.176214, 0.349659, 0.329720, 0.266154, 0.645259]` | `2.9844` | EI | Slight decrease |
| F8 | `[0.118399, 0.068433, 0.135710, 0.092100, 0.390808, 0.777446, 0.481396, 0.876292]` | `9.6625` | UCB | Slight decrease |

---

## Summary

Week 12 delivered three new records and confirmed the value of the boundary
exploitation strategy for Function 5.

**Function 5** achieved the most dramatic result of the entire campaign:
**8662.40**, nearly doubling the previous best of 4478.77. Fixing all four
coordinates to 0.999999 — based on nine weeks of evidence pointing to the
upper boundary as the optimum region — produced a result that had been
consistently approached but never reached under the boundary-pinned strategy
where x1 was left free. This confirms that the true optimum for Function 5
lies at or very near the corner of the search space.

**Function 3** set a new best at **-0.0088**, its strongest result across
the entire 12-week campaign. The EI-driven single-centre strategy with
very tight localStd=0.004 successfully refined around the region established
in Week 9, continuing the steady improvement trajectory that has characterised
this function throughout the campaign.

**Function 6** also achieved a new best at **-0.4130**, its strongest result
since the campaign began. After repeated failures across different acquisition
strategies, the UCB approach with moderate exploration in the final two weeks
identified a more productive region than any previously sampled.

**Function 2** returned **0.6150**, slightly below its best of 0.663. The
suggestion remained within the established high-value region near (0.70, 0.93),
confirming that the surrogate model continues to target the correct area even
if the precise peak was not hit this round.

**Function 7** returned **2.9844**, slightly below its Week 10 peak of 3.080.
The surrogate model operated in a consistent region throughout the final weeks,
and the small variation confirms that the landscape near the optimum is smooth
and relatively broad.

**Function 8** returned **9.6625**, marginally below its best of 9.675.
The switch to UCB with wider local sampling in the final two weeks produced
a slightly different region of the 8-dimensional space, which did not
outperform the confirmed peak. With only one query remaining, the surrogate
could not fully refine this high-dimensional landscape.

**Function 4** returned **0.1929**, below its best of 0.482. The GP
surrogate for this function never achieved reliable calibration — sigma
values consistently above 3.0 throughout the campaign indicate that the
landscape violates the smoothness assumptions of the RBF kernel. The Week 8
peak remains the strongest result and was never reproducibly approached.

**Function 1** again produced a near-zero output, as it has throughout the
entire campaign. With no identifiable cluster or peak region despite 12
rounds of querying, this function either contains an extremely narrow
optimum that was never sampled, or is largely flat across the accessible
region of the search space.

---

## Final Campaign Summary

| Function | Dim | W1 Best | Final Best | Overall Change |
|----------|-----|---------|------------|----------------|
| F1 | 2 | ~0 | ~0 | No improvement |
| F2 | 2 | 0.641 | 0.663 | +3.4% |
| F3 | 3 | -0.483 | -0.009 | Major improvement |
| F4 | 4 | -31.18 | +0.482 | Major improvement |
| F5 | 4 | 1163.7 | 8662.4 | +644% |
| F6 | 5 | -2.75 | -0.413 | Major improvement |
| F7 | 6 | 2.27 | 3.080 | +35.7% |
| F8 | 8 | 9.31 | 9.675 | +3.9% |

Six out of eight functions showed clear improvement over the 12-week campaign.
The most significant gains occurred in Function 5 (+644%) and in functions
that started from strongly negative values (F3, F4, F6). Functions 7 and 8
showed steady, stable improvement consistent with smooth, well-calibrated
landscapes. Functions 1 and 4 remained the most resistant to systematic
improvement, likely due to landscape characteristics that the GP surrogate
could not reliably model within the available query budget.

The campaign demonstrated that adaptive, function-specific Bayesian
optimisation consistently outperforms uniform strategies — every major
improvement across the 12 weeks was the result of a targeted configuration
change informed by accumulated observations rather than a generic rule
applied across all functions.

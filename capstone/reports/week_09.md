# Week 9 — Tight Exploitation and Continued Refinement

## Overview

Week 9 marked the transition into the final phase of the optimisation campaign,
with four weeks remaining and a clear focus on tight local exploitation.

Following the mixed results of Week 8, candidate counts were increased to 150,000
to enable finer search resolution. The strategy shifted almost entirely to
Expected Improvement across all functions, with UCB retained only where
surrogate uncertainty remained too high for EI to operate reliably.

Key adjustments included:
- Increased candidate pool to 150k for higher search resolution
- Switched F7 and F8 from UCB to EI for stable local refinement
- Tightened local sampling for F3 and F4 around confirmed best regions
- Reduced UCB kappa for F2 to limit excursions from the high-value region
- F4 moved to UCB with low kappa to manage high surrogate uncertainty
- Continued EI exploitation for F5 near the upper search boundary

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Change vs Week 8 |
|----------|----------------|--------|------------------|------------------|
| F1 | `[0.008366, 0.696434]` | `~0` | Spread | No meaningful change |
| F2 | `[0.703801, 0.903428]` | `0.5997` | UCB | Improvement |
| F3 | `[0.417031, 0.661276, 0.519612]` | `-0.0106` | EI | New best |
| F4 | `[0.335328, 0.406273, 0.414167, 0.382835]` | `0.2506` | UCB | Decrease |
| F5 | `[0.070114, 0.999999, 0.953461, 0.999999]` | `3858.50` | EI | Decrease |
| F6 | `[0.762241, 0.254723, 0.628707, 0.856341, 0.080725]` | `-0.5795` | EI | Slight improvement |
| F7 | `[0.183748, 0.179606, 0.285265, 0.288742, 0.257044, 0.707412]` | `2.8999` | EI | Slight decrease |
| F8 | `[0.143573, 0.063572, 0.105755, 0.099492, 0.391971, 0.802965, 0.452828, 0.860347]` | `9.6748` | EI | New best |

---

## Summary

Week 9 produced two new records and confirmed the value of tighter local
exploitation in the final phase of the campaign.

**Function 3** achieved a new best of **-0.0106**, continuing its steady upward
trend under EI-driven refinement. The improvement from -0.0168 to -0.0106
suggests that the surrogate model is increasingly well calibrated in this
region, and further gains remain plausible in the remaining weeks.

**Function 8** set a new best at **9.6748**, recovering from the regression
observed in Week 8. Switching from UCB to EI allowed the model to concentrate
sampling around the confirmed high-value region rather than exploring further
afield in the 8-dimensional space. This reinforces the view that EI is the
more appropriate strategy at this stage of the campaign.

**Function 2** improved to **0.5997**, its strongest result since Week 1 and
the closest it has come to the original best of 0.641. The reduced UCB kappa
and tight single-centre sampling directed the query back toward the established
high-value region near (0.71, 0.93), confirming that conservative exploitation
is the correct approach for this landscape.

**Function 5** regressed from 4440.77 to **3858.50** despite operating in the
same boundary region. This suggests that small deviations in the x3 coordinate
from 0.999999 can produce significant output variation, indicating a narrow
ridge rather than a broad plateau near the optimum. Future queries will
maintain x2, x3 and x4 as close to the boundary as possible.

**Function 4** decreased from 0.4821 to **0.2506**. High surrogate uncertainty
in this region (sigma ≈ 3.3) continues to make consistent exploitation
difficult. The landscape appears to contain a narrow positive region that
the GP struggles to model reliably with the available data.

**Function 7** remained stable at **2.8999**, slightly below its Week 7 peak
of 2.96. The surrogate model continues to operate in a consistent region,
and the switch to EI did not produce a regression. Tighter localStd in
subsequent weeks may help close the gap to the peak.

**Function 6** showed a slight improvement to **-0.5795**, recovering partially
from the Week 8 result. EI appears more stable than UCB for this landscape,
and will be retained going forward.

**Function 1** again produced a near-zero output. With four weeks remaining
and no meaningful signal discovered after nine rounds, this function is
unlikely to yield further improvement. Spread-based sampling will continue
as a low-cost exploratory strategy while effort is concentrated elsewhere.

Overall, Week 9 confirmed that the campaign is entering its final refinement
phase. The priority for the remaining four weeks is to consolidate and
extend the records set in F3, F7 and F8, recover F2 to its Week 1 peak,
and maintain F5 at its upper boundary region.
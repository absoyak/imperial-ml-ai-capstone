# Week 4 — Stabilisation and Function-Specific Tuning

## Overview

Week 4 focused on stabilisation and controlled exploitation.

After the mixed results of Week 3, particularly the drift observed in Function 2,
this iteration introduced explicit function-specific tuning rather than relying
on a uniform Gaussian Process configuration.

Key changes included:
- Target normalisation before GP fitting
- Adjusted kernel length scale and noise level for sensitive functions
- More concentrated local candidate sampling
- Controlled UCB-based exploitation where necessary

The objective was to reduce instability while preserving exploration where beneficial.

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Change vs Week 3 |
|----------|----------------|--------|------------------|------------------|
| F1 | `[0.000934, 0.005257]` | `~0` | Variance | No change |
| F2 | `[0.598126, 0.609009]` | `0.1169` | UCB | Decrease |
| F3 | `[0.386745, 0.401850, 0.488207]` | `-0.0237` | EI | Improvement |
| F4 | `[0.452635, 0.521431, 0.453934, 0.329808]` | `-2.08` | UCB | Improvement |
| F5 | `[0.071211, 0.851797, 0.913154, 0.995216]` | `2100.97` | EI | Major improvement |
| F6 | `[0.709318, 0.290415, 0.606401, 0.757587, 0.058188]` | `-0.4761` | EI | Improvement |
| F7 | `[0.023913, 0.307314, 0.284348, 0.229396, 0.298611, 0.655896]` | `2.41` | EI | Improvement |
| F8 | `[0.069838, 0.058892, 0.111458, 0.095078, 0.279214, 0.807952, 0.455800, 0.839821]` | `9.62` | UCB | Slight improvement |

---

## Summary

Week 4 delivered the strongest overall improvement so far.

Six out of eight functions improved relative to Week 3.
The most significant gain occurred in Function 5, which increased from 1418.63 to 2100.97,
confirming that controlled exploitation in stable high-value regions is highly effective.

Functions F6 and F4 also recovered substantially, suggesting that the revised
kernel and sampling adjustments improved surrogate calibration.

Function 7 continued its steady upward trend, reinforcing confidence in that region.
Function 8 achieved a small but consistent improvement.

Function 2 remains unstable. Despite targeted tuning and controlled UCB exploitation,
performance did not return to its Week 1 peak. This suggests either:
- A very narrow optimum region
- Surrogate miscalibration
- Or that the initial result may already be near the global maximum

Overall, Week 4 marks a transition from exploratory experimentation
to stabilised, adaptive optimisation. The strategy is now clearly function-aware,
with hyperparameters and acquisition behaviour adjusted per landscape.

High-dimensional functions (6D–8D) are showing increasing stability,
indicating that the approach scales better than earlier iterations suggested.

The optimisation process is becoming more disciplined and less reactive.
Future iterations will likely focus on:
- Tight local refinement in confirmed strong regions (F5, F7, F8)
- Targeted recovery attempts in F2
- Controlled exploration only where uncertainty meaningfully remains
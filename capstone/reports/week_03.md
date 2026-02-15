# Week 3 — Hybrid Strategy and Recovery Attempts

## Overview

Week 3 combined targeted exploitation and renewed exploration.
Based on Week 2 results, the strategy aimed to recover performance in functions that had declined,
while further refining regions that showed consistent strength.

Rather than applying a single acquisition rule across all functions,
this round used a function-specific hybrid approach informed by previous outcomes.

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Change vs Week 2 |
|--------|----------------|--------|------------------|------------------|
| F1 | `[0.999278, 0.996924]` | `~0` | Variance | No change |
| F2 | `[0.596669, 0.999882]` | `0.1498` | EI | Significant decrease |
| F3 | `[0.999688, 0.003372, 0.753921]` | `-0.1727` | EI | Slight decrease |
| F4 | `[0.014237, 0.923067, 0.992721, 0.282077]` | `-32.65` | Variance | Improvement |
| F5 | `[0.156695, 0.840579, 0.897269, 0.927818]` | `1418.63` | EI | Strong improvement |
| F6 | `[0.924078, 0.013058, 0.044861, 0.018634, 0.984235]` | `-3.06` | Variance | Performance worsened |
| F7 | `[0.049729, 0.346230, 0.244229, 0.245910, 0.288484, 0.677631]` | `2.33` | EI | Improvement |
| F8 | `[0.186164, 0.682961, 0.392368, 0.441024, 0.760831, 0.740775, 0.346851, 0.262693]` | `9.30` | EI | Slight improvement |

---

## Summary

Week 3 delivered mixed but meaningful results.
Four functions improved (F4, F5, F7, F8), with F5 achieving the strongest performance so far.
F7 also exceeded its previous best, confirming stability in that region.

However, F2 and F6 regressed, suggesting that the surrogate model may still be miscalibrated
in certain landscapes, especially where output scales differ significantly or data remains sparse.

This round reinforced an important insight: uniform strategies are less effective than adaptive,
function-specific decision-making.

The optimisation process is becoming more structured, but uncertainty remains high in higher dimensions.
Future iterations will likely require tighter exploitation in stable regions and more disciplined
exploration in unstable ones.

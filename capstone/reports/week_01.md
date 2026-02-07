# Week 1 — Initial Exploration & Cherry-Picked Strategy

## Overview

In Week 1, the main objective was to explore the black-box functions under severe data scarcity.
Multiple acquisition functions (UCB, Variance, and Probability of Improvement) were evaluated for
each function, and the final submission was **cherry-picked** based on predicted performance,
uncertainty, and qualitative judgement.

The focus was on understanding the response landscapes rather than maximising performance
immediately.

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Strategy Notes |
|--------|----------------|--------|------------------|----------------|
| F1 | `[0.004001, 0.998839]` | `0` | Mixed (Cherry-picked) | Flat response, little signal |
| F2 | `[0.713435, 0.935648]` | `0.6410` | PI | Strong early indication |
| F3 | `[0.541312, 0.999541, 0.998621]` | `-0.4832` | Variance | High uncertainty |
| F4 | `[0.008874, 0.372953, 0.921276, 0.970844]` | `-31.18` | Variance | Large negative values |
| F5 | `[0.208101, 0.841062, 0.884248, 0.894119]` | **`1163.73`** | UCB / PI | Very strong region identified |
| F6 | `[0.868133, 0.025293, 0.985287, 0.006390, 0.933349]` | `-2.75` | Variance | Noisy, exploratory |
| F7 | `[0.034259, 0.346213, 0.256993, 0.270013, 0.352658, 0.664671]` | `2.27` | UCB | Promising region |
| F8 | `[0.456175, 0.098687, 0.218244, 0.634420, 0.290337, 0.299393, 0.204119, 0.739028]` | `9.31` | UCB | Sparse high-dimensional data |

---

## Summary

Week 1 prioritised exploration and uncertainty reduction.  
The cherry-picking approach allowed comparison across acquisition functions and provided
initial insights into which functions might benefit from early exploitation versus continued exploration.

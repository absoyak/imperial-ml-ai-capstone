# Week 2 — Refinement and Targeted Exploration

## Overview

Week 2 built upon the expanded dataset from Week 1.
The strategy shifted towards **selective exploitation** for functions that showed early promise,
while maintaining exploration for high-dimensional or poorly understood functions.

Decisions were informed by model predictions, uncertainty estimates, and reflection on Week 1 outcomes.

---

## Selected Queries and Results

| Function | Selected Input | Output | Acquisition Used | Change vs Week 1 |
|--------|----------------|--------|------------------|------------------|
| F1 | `[0.995368, 0.000958]` | `0` | Variance | No change |
| F2 | `[0.717468, 0.938650]` | `0.5398` | PI | Slight decrease |
| F3 | `[0.007631, 0.980377, 0.747038]` | `-0.1472` | Variance | Improvement |
| F4 | `[0.562919, 0.982826, 0.985459, 0.012742]` | `-34.75` | Variance | Performance worsened |
| F5 | `[0.211735, 0.803622, 0.860977, 0.935169]` | `1128.36` | PI | Slight decrease |
| F6 | `[0.009728, 0.022631, 0.062477, 0.071256, 0.002874]` | `-2.01` | Variance | Improvement |
| F7 | `[0.015810, 0.376645, 0.192640, 0.296911, 0.398773, 0.696160]` | `1.80` | PI | Slight decrease |
| F8 | `[0.136117, 0.113665, 0.438758, 0.324447, 0.101986, 0.795160, 0.428196, 0.688728]` | `9.24` | PI | Stable |

---

## Summary

Week 2 demonstrated the difficulty of optimisation with limited samples, particularly in
high-dimensional spaces. While some functions showed improvement (e.g. F3, F6),
others highlighted the risk of early exploitation.

Overall, this week reinforced the need for adaptive strategies that balance exploration and
exploitation based on dimensionality and observed behaviour.

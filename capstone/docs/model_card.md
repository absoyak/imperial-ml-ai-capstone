# Model Card: Gaussian Process Black-Box Optimiser

## Overview

**Name:** GP-BBO (Gaussian Process Black-Box Optimiser)  
**Type:** Surrogate-based sequential optimisation  
**Version:** Week 11 (final exploitation configuration)  
**Author:** Burak Absoy  
**Repository:** `https://github.com/absoyak/imperial-ml-ai-capstone`

---

## Intended Use

**Suitable for:**
- Sequential optimisation of unknown, expensive-to-evaluate functions
- Low-dimensional input spaces (2D–8D) with limited evaluation budgets
- Functions with smooth, continuous response surfaces
- Settings where gradient information is unavailable

**Not suitable for:**
- Noisy or stochastic functions (the GP assumes near-deterministic outputs)
- Very high-dimensional spaces (>10D) with fewer than 50 observations
- Discontinuous or highly multimodal landscapes
- Time-critical applications (GP inference does not scale to large datasets)

---

## Model Details

**Surrogate model:** Gaussian Process Regression  
**Kernel:** Radial Basis Function (RBF / squared exponential)  
- Base length scale: 0.2 (adjusted to 0.6 for Function 2)
- Noise level: 1e-8 (adjusted to 1e-6 for Function 2)

**Posterior computation:** Cholesky decomposition with Jitter regularisation

**Acquisition functions used across the campaign:**
- Upper Confidence Bound (UCB): exploration-heavy, controlled by kappa
- Variance: pure uncertainty reduction
- Probability of Improvement (PI): conservative exploitation
- Expected Improvement (EI): balanced exploitation, primary strategy from Week 5

**Candidate generation:** 150,000 random candidates per query  
- Local candidates: Gaussian noise around top-k observed points
- Global candidates: uniform random across [0, 1)^d
- Local fraction and standard deviation tuned per function

**Y-normalisation:** Z-score normalisation applied to outputs before GP fitting

---

## Strategy Evolution Across Rounds

**Weeks 1–2:** Uniform exploration using UCB and Variance acquisition across all
functions. Cherry-picking between acquisition functions based on initial signal.

**Week 3:** Hybrid strategy introduced — EI for functions showing promise,
Variance for poorly understood landscapes.

**Week 4:** Function-specific hyperparameter tuning. Y-normalisation activated.
Local candidate sampling introduced around top-performing regions.

**Week 5:** GP inference stabilised. Scipy-accelerated normal CDF. Dead
normalisation code fixed. F2 surrogate miscalibration identified and addressed
by restricting to single-centre sampling with larger length scale.

**Week 6:** Spread-based acquisition introduced for F1 after repeated zero
outputs. F4 improved dramatically using UCB exploration.

**Weeks 7–8:** Acquisition portfolio narrowed toward EI. UCB retained only
for functions with high surrogate uncertainty. Local standard deviations
tightened progressively.

**Weeks 9–10:** Full exploitation phase. All functions switched to EI.
F5 received custom boundary-pinned candidate generation — x2/x3/x4 fixed
near 0.999999 based on observed ridge structure. topK reduced to 1 for
functions with a single confirmed high-value region.

**Week 11:** Maximum exploitation configuration. localStd reduced further
for F2, F3, F7 and F8. topK=1 applied to F1, F2, F3, F4 and F8.

---

## Performance Summary

Results as of Week 11 (cumulative best per function):

| Function | Dim | Cumulative Best | Trend |
|----------|-----|-----------------|-------|
| F1 | 2 | ~0 | Flat throughout |
| F2 | 2 | 0.6632 | W10 new record — exceeded W1 peak |
| F3 | 3 | -0.0106 | Steady improvement |
| F4 | 4 | 0.4821 | Volatile, narrow peak |
| F5 | 4 | 4440.77 | Strong — near plateau |
| F6 | 5 | -0.4761 | Resistant to improvement |
| F7 | 6 | 3.0801 | W10 new record |
| F8 | 8 | 9.6748 | Stable improvement |

**Functions with consistent improvement:** F3, F5, F7, F8  
**Functions that plateaued early:** F2, F5 (diminishing returns from Week 8)  
**Functions that resisted improvement:** F1, F6  
**Functions with high surrogate uncertainty:** F4

---

## Assumptions and Limitations

**Assumptions:**
- Functions are smooth and continuous — the RBF kernel assumes nearby inputs
  produce similar outputs. Violations of this assumption (as observed in F4)
  cause the surrogate to produce unreliable predictions.
- Functions are deterministic — the same input always returns the same output.
  A very small noise level (1e-8) is added for numerical stability only.
- The global optimum lies within [0, 1)^d — no transformation of the input
  space is applied.

**Limitations:**
- With 19–21 observations per function, the GP is severely data-starved in
  higher dimensions (6D, 8D). The surrogate cannot reliably model the full
  landscape and may produce overconfident predictions far from observed points.
- The RBF kernel with fixed length scale cannot adapt to functions with
  varying smoothness across regions. Automatic length scale optimisation via
  marginal likelihood was not implemented.
- Candidate generation is stochastic — results vary across runs. No random
  seed is set, so reproducibility requires fixing numpy's RNG state.
- Local sampling introduces a sampling bias toward previously observed
  high-value regions. In later weeks, the strategy is effectively greedy
  and may miss global optima far from the current best.

---

## Ethical Considerations

This model is used exclusively for academic optimisation within a controlled
programme environment. The functions being optimised are synthetic and have
no real-world impact.

**Transparency:** All query decisions are logged in weekly markdown reports
in the `reports/` directory. Each report documents the acquisition function
used, the suggested input, the observed output, and the reasoning behind
the strategic choice. This makes the decision-making process fully auditable.

**Reproducibility:** The full codebase, weekly scripts, and dataset are
publicly available in the GitHub repository. A researcher with access to
the same initial seed data and weekly outputs could reproduce all results
by running the weekly scripts in sequence.

**Bias:** The dataset is biased toward regions that produced high outputs
in earlier weeks. This is an intentional consequence of the exploitation
strategy and is acknowledged as a limitation for any secondary analysis
of the data.

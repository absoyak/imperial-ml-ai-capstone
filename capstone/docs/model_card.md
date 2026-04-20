# Model Card: GP-BBO (Gaussian Process Black-Box Optimiser)

## Model Description

**Input:** A d-dimensional vector x = [x1, x2, ..., xd] where each xi ∈ [0, 0.999999],
specified to six decimal places. Dimensionality d ranges from 2 to 8 depending on
the function being optimised. The model takes the full history of previously observed
(input, output) pairs as context for each query decision.

**Output:** A single suggested query point — a d-dimensional vector formatted as
`x1-x2-...-xd` for submission to the BBO portal. The model also outputs the GP
posterior mean (μ) and standard deviation (σ) at the suggested point, which serve
as diagnostic indicators of surrogate confidence.

**Model Architecture:**

The system is a Bayesian optimisation framework built around a Gaussian Process
surrogate model implemented from scratch in NumPy.

- **Surrogate:** Gaussian Process Regression with RBF (squared exponential) kernel.
  Posterior mean and variance are computed via Cholesky decomposition for numerical
  stability, with jitter regularisation as a fallback.
- **Kernel:** RBF with per-function length scales (base: 0.2; F2: 0.6; F3: 0.35;
  F4: 0.30; F6: 0.28; F7/F8: 0.22) and noise levels (base: 1e-8; F2: 1e-6).
- **Y-normalisation:** Z-score normalisation applied to outputs before GP fitting
  to stabilise inference across functions with very different output scales.
- **Acquisition functions:** Expected Improvement (EI), Upper Confidence Bound (UCB),
  Probability of Improvement (PI), Variance sampling, and Spread-based maximin sampling.
  EI became the primary acquisition function from Week 5 onwards.
- **Candidate generation:** 150,000–200,000 candidates per query, mixing local
  candidates (Gaussian noise around top-k observed points) and global candidates
  (uniform random). Local fraction and standard deviation are tuned per function.
  Function 5 uses fixed boundary submission at (0.999999, 0.999999, 0.999999, 0.999999)
  based on accumulated ridge evidence.
- **filterTopK:** From Week 12, the GP is fitted only on the top-k most recent
  high-value observations for selected functions (F2, F6 in Week 12; extended to
  F7 in Week 13), reducing noise from early exploratory queries.

---

## Performance

Performance is measured as the cumulative best output observed per function
across all 13 weeks of sequential queries. Higher is better for all functions.

| Function | Dim | Week 1 Best | Final Best (W13) | Improvement |
|----------|-----|-------------|------------------|-------------|
| F1 | 2 | ~0 | ~0 | None |
| F2 | 2 | 0.641 | 0.663 | +3.4% |
| F3 | 3 | -0.483 | -0.009 | Major |
| F4 | 4 | -31.18 | +0.482 | Major |
| F5 | 4 | 1163.7 | 8662.4 | +644% |
| F6 | 5 | -2.75 | -0.413 | Major |
| F7 | 6 | 2.27 | 3.181 | +40.1% |
| F8 | 8 | 9.31 | 9.675 | +3.9% |

**Functions with consistent improvement:** F3, F5, F7, F8
**Functions that plateaued or were volatile:** F2 (narrow peak), F4 (irregular landscape)
**Functions that resisted improvement:** F1, F6

The most significant gains occurred in F5 (+644%), where the optimum was
eventually located at the corner of the search space (all coordinates near
0.999999). F3, F4 and F6 showed major improvements from strongly negative
starting points. F7 delivered steady gains throughout the campaign, reaching
a new best of 3.181 in the final round after the introduction of filterTopK
for this function.

---

## Limitations

- **Data scarcity in high dimensions:** With approximately 23 observations per
  function, the GP is severely under-informed in 6D and 8D spaces. The surrogate
  cannot reliably model the full landscape and may produce overconfident predictions
  far from observed points.
- **Fixed kernel length scale:** The RBF kernel with a fixed length scale cannot
  adapt to functions with varying smoothness across regions. Automatic length scale
  optimisation via marginal likelihood was not implemented, meaning hyperparameters
  are set manually based on observed behaviour rather than learned from data.
- **Stochastic candidate generation:** Results vary across runs as no random seed
  is set. Reproducibility requires fixing NumPy's RNG state before running.
- **Local sampling bias:** In later weeks, the strategy becomes effectively greedy,
  concentrating all candidates near previously observed high-value regions. This
  may miss global optima located far from the current best.
- **Single query per week per function:** The one-query-per-week constraint severely
  limits the rate of learning, particularly in higher-dimensional functions where
  the surrogate requires more observations to become reliable.
- **Unmodellable landscapes:** Function 4 consistently produced high surrogate
  uncertainty (sigma > 3.0) across all 13 weeks, indicating that the RBF smoothness
  assumption does not hold for this function. Function 1 produced near-zero outputs
  in every round, suggesting either an extremely narrow peak that was never sampled
  or a largely flat landscape in the accessible region.

---

## Trade-offs

**Exploration vs exploitation:** High-kappa UCB in early weeks maximised
uncertainty reduction but occasionally directed queries to unproductive regions
(notably F2 in Week 8, where UCB caused a sharp performance drop). Switching to
EI from Week 5 onwards improved stability but increased the risk of premature
convergence to local optima.

**Surrogate complexity vs reliability:** A manually implemented NumPy GP
provides full control over hyperparameters and candidate generation but lacks
automatic kernel selection or marginal likelihood optimisation. Functions like
F4, where the landscape violated the smoothness assumption, could not be
modelled reliably regardless of parameter tuning.

**Local fraction vs global coverage:** High localFrac (0.95–1.00) in final
weeks delivered precise refinement for well-understood functions (F2, F3, F7)
but eliminated any chance of discovering better regions elsewhere. For functions
with poorly understood landscapes (F1, F6), this trade-off was particularly costly.

**filterTopK:** Fitting the GP on a subset of high-value observations (rather
than all historical data) improved surrogate calibration in the local region
but discarded potentially useful global structure learned from early queries.
In Week 13, extending filterTopK to F7 produced the campaign's final record
for that function, validating the approach for high-dimensional smooth landscapes.

**Ethical Considerations:**

This model is used exclusively for academic optimisation within a controlled
programme environment. The functions being optimised are synthetic and have
no real-world impact. All query decisions are logged in weekly markdown reports
in the `reports/` directory, making the decision-making process fully auditable.
The full codebase and dataset are publicly available at
`https://github.com/absoyak/imperial-ml-ai-capstone`.

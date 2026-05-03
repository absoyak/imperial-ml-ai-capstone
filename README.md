# Black-Box Optimisation (BBO) Capstone Project

## Project Overview

This project addresses a constrained black-box optimisation problem in which objective functions are unknown and can only be evaluated through limited, expensive queries.

The goal is to maximise each function efficiently while minimising evaluations.  
The setup mirrors real-world ML deployment scenarios:

- Expensive evaluations
- Unknown functional structure
- Sparse and noisy observations
- Limited query budgets

Weekly progress is documented in:

- [Week 01](capstone/reports/week_01.md)
- [Week 02](capstone/reports/week_02.md)
- [Week 03](capstone/reports/week_03.md)
- [Week 04](capstone/reports/week_04.md)
- [Week 05](capstone/reports/week_05.md)
- [Week 06](capstone/reports/week_06.md)
- [Week 07](capstone/reports/week_07.md)
- [Week 08](capstone/reports/week_08.md)
- [Week 09](capstone/reports/week_09.md)
- [Week 10](capstone/reports/week_10.md)
- [Week 11](capstone/reports/week_11.md)
- [Week 12](capstone/reports/week_12.md)
- [Week 13](capstone/reports/week_13.md)

---

- [Dataset Datasheet](capstone/docs/datasheet.md)
- [Model Card](capstone/docs/model_card.md)
- [Method Summary Notebook](capstone/notebooks/method_summary.ipynb)

---

## Results

### Per-Function Progress

![Per-function optimisation progress](capstone/data/bbo_progress_per_function.png)

### Normalised Cumulative Best (All Functions)

![Normalised cumulative best across all functions](capstone/data/bbo_progress_normalised.png)

---

## Problem Setting

Each function receives an input vector:

x1, x2, ..., xn

Where:
- xi ∈ [0, 1)
- Six decimal precision
- Dimensionality ranges from 2D to 8D

The output is a scalar value to be maximised.

Constraints include:

- Unknown function form
- Strongly varying output magnitudes
- Severe sparsity in higher dimensions
- One query per function per week (12 weeks total)

---

## Theoretical Foundations

The optimisation framework is based on **Bayesian Optimisation** using a Gaussian Process (GP) surrogate model.

Key references informing this design:

- Rasmussen & Williams (2006) — *Gaussian Processes for Machine Learning*
- Jones et al. (1998) — *Efficient Global Optimization (EGO)*
- Srinivas et al. (2010) — *Gaussian Process Upper Confidence Bound (GP-UCB)*

Bayesian optimisation is particularly suitable for:

- Expensive black-box functions
- Low to moderate data regimes
- Problems requiring uncertainty-aware decisions

The surrogate model approximates the unknown function, while acquisition functions guide exploration vs exploitation.

---

## Surrogate Model

A Gaussian Process with an RBF (squared exponential) kernel is used.

Key implementation features:

- Cholesky decomposition for numerical stability
- Target normalisation (z-score) prior to GP fitting
- Function-specific kernel length scales
- Selective noise regularisation
- Explicit variance clipping for stability

The GP was implemented manually (NumPy-based) rather than using scikit-learn to:

- Retain full control over acquisition behaviour
- Modify hyperparameters per function
- Improve interpretability of surrogate dynamics

---

## Acquisition Strategies

Multiple acquisition functions are implemented:

- Expected Improvement (EI)
- Upper Confidence Bound (UCB)
- Variance sampling
- Spread-based maximin sampling (Week 6 addition)

Each function uses a tailored acquisition configuration.

Exploration–exploitation balance is adjusted per landscape:

- EI for aggressive local refinement
- UCB for uncertainty-aware recovery
- Variance for exploration-dominant cases
- Spread sampling for flat/degenerate surfaces (Function 1)

---

## Strategy Evolution

The optimisation strategy evolved in three main phases:

- **Exploration phase:** Broad sampling using multiple acquisition functions to understand the global structure.
- **Hybrid phase:** Function-specific tuning and mixed exploration–exploitation strategies as patterns emerged.
- **Exploitation phase:** Tight local refinement around confirmed high-performing regions, including boundary exploitation and Top K filtering.

Detailed week-by-week decisions are documented in the reports section.

---

## Design Trade-offs

This project explicitly balances:

- Exploration vs exploitation
- Surrogate confidence vs model overconfidence
- Stability vs aggressive peak-seeking
- Global coverage vs trust-region refinement

High-dimensional functions (6D–8D) required tighter local refinement and controlled uncertainty handling.

---

## Data Availability

The original weekly query results and intermediate `.npy` datasets are not included in this repository, as they were generated through the course platform.

The repository focuses on documenting the optimisation process, code, reports, and final results. Reproducing exact numerical results requires access to the original query outputs.

---

## Future Extensions

Potential next steps include:

- Neural network surrogate models
- Trust-region Bayesian Optimisation (TuRBO)
- Random embeddings for high-dimensional BO
- Automatic hyperparameter optimisation of the surrogate
- Comparative benchmarking against scikit-learn GP

---

## Reflection

This project demonstrates decision-making under uncertainty rather than brute-force optimisation. The strategy evolved from a generic Bayesian optimisation setup into a function-aware process driven by observed behaviour.

Strong improvements in F5 and F7 confirmed that tight local refinement and boundary exploitation are highly effective when structure exists. In contrast, functions such as F1 showed that not all problems are learnable within limited query budgets, highlighting the limitations of surrogate-based approaches.

A key takeaway is that no single acquisition strategy works across all functions. Performance improved only when each function was treated individually and adjusted based on observed patterns rather than relying on a fixed global configuration.

Late-stage adjustments, such as fixing F5 to the boundary and applying filtered training sets for F7, showed that meaningful gains are still possible even in the final iterations. This reinforces the importance of continuous analysis rather than assuming convergence too early.

---

## Final Strategy Reflection

The most effective strategies were simple but adaptive.

Key elements were:
* Tight local refinement after identifying reliable regions
* Expected Improvement for stable landscapes
* Top K filtering to reduce noise from early exploration
* Function-specific tuning instead of a single global setup
* Boundary locking for Function 5

The project also showed the limits of Gaussian Processes. Some functions were learnable, while others remained unstable despite multiple strategies.

The main lesson is that optimisation depends on recognising structure. A strong strategy is not the most complex one, but the one that adapts when the model is no longer reliable.

---

## Non-technical Summary

This project explores how to optimise unknown systems using very limited information. Instead of knowing how a function works, we can only test it by trying different inputs and observing the results.

Over 13 weeks, I used a machine learning approach called Bayesian optimisation to gradually improve performance. The system learns from past results and decides where to try next, balancing exploration of new areas and refinement of known good regions.

Some functions showed clear patterns and were successfully optimised, while others remained unpredictable. This reflects real-world scenarios where not all problems are equally learnable.

---

## Reproducibility

To reproduce results, run the weekly scripts in the `capstone/src` folder. 
The provided scripts demonstrate the full optimisation logic and decision pipeline.
Note that results may slightly vary due to stochastic candidate generation 
unless a random seed is fixed.

The full project implementation, weekly reports, datasheet, model card, method summary notebook, and final visualisations are available on GitHub.

https://github.com/absoyak/imperial-ml-ai-capstone

# Black-Box Optimisation (BBO) Capstone Project

## Project Overview

This project tackles a black-box optimisation problem where the objective functions are unknown and can only be evaluated through limited queries. The goal is to maximise each function efficiently while keeping the number of evaluations low.

The challenge reflects real-world ML scenarios: expensive evaluations, unknown structure, sparse data, and the need to adapt strategy iteratively rather than search blindly.

Weekly progress is documented in:
- reports/week_01.md
- reports/week_02.md
- reports/week_03.md
- reports/week_04.md (to be updated after results)

---

## Inputs and Outputs

Each function receives an input vector:

x1 – x2 – ... – xn

where:
- xi ∈ [0, 1)
- Values are specified to six decimal places
- Dimensionality ranges from 2D to 8D

The output is a scalar value.  
The objective is to maximise this value using a limited number of queries.

Main constraints:
- Unknown function structure
- Strongly varying output scales
- Severe sparsity in higher dimensions

---

## Objective

The aim is to maximise all eight functions with minimal queries. Since the functions are black-box, all decisions rely on surrogate modelling and acquisition strategies.

Key challenges include:
- Curse of dimensionality (especially 6D–8D)
- Balancing exploration and exploitation
- Avoiding overfitting with limited data

---

## Technical Approach

Week 1:  
Implemented a Gaussian Process (RBF kernel) and tested multiple acquisition strategies (UCB, Variance, PI). Cherry-picked the strongest suggestions per function to establish a baseline.

Week 2:  
Refined the GP model and tailored acquisition strategies per function. Exploited promising regions (e.g., F2, F7) and explored uncertain ones (e.g., F4, F6). This week highlighted scale differences and sparsity issues in high dimensions.

Week 3:  
Adopted a hybrid strategy:
- Re-exploit strong candidates that slightly regressed
- Increase exploration in unstable or poorly understood regions
- Use Expected Improvement for refinement
- Use variance-driven sampling where uncertainty dominates

The approach is now function-specific rather than uniform.

Week 4:  
Introduced function-specific stabilisation and hyperparameter refinement.

Key updates:
- Applied target normalisation (z-score) before GP fitting for numerical stability.
- Adjusted kernel length scale and noise level selectively per function.
- Introduced controlled exploitation using UCB for unstable functions (notably F2).
- Increased local candidate concentration around top-performing regions.
- Reduced aggressive exploration in sensitive functions.

Function 2 required dedicated handling after performance drift in Weeks 2–3.  
Stabilisation steps included:
- Custom kernel length scale
- Increased noise level
- Moderate UCB kappa
- Concentrated local sampling

The strategy is now explicitly adaptive per function rather than globally configured.


---

## SVM Perspective

Although the optimisation framework is GP-based, SVM concepts from Module 14 influence the strategy. A soft-margin SVM could classify high- vs low-performance regions and help restrict the search space before regression refinement. Kernel methods are particularly relevant for non-linear response surfaces.

A hybrid SVM + GP approach may become useful as more data accumulates.

---

## Reflection

This project is less about finding the perfect model and more about structured decision-making under uncertainty. Each round forces reassessment of assumptions and strategic adjustment.

The focus is disciplined iteration: model, test, adapt.

The code stays simple. The strategy evolves.

Week 4 marked a transition from generic Bayesian optimisation to adaptive, function-aware optimisation. Stability and controlled exploitation became as important as exploration, especially in higher dimensions.

## Neural Network Perspective

Although the current surrogate remains Gaussian Process-based, insights from neural networks and backpropagation influence the strategy. 

Hyperparameter sensitivity (learning rate, model capacity, regularisation) reinforced the importance of stability and controlled updates in optimisation.

Future iterations may explore neural network surrogates with gradient-informed query steps, particularly for higher-dimensional functions (6D–8D).


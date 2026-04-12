# Datasheet: BBO Capstone Project Dataset

## Motivation

This dataset was created to support a black-box optimisation (BBO) challenge
as part of the Imperial College London Machine Learning and AI Professional
Certificate programme. The dataset records all query inputs and corresponding
function evaluation outputs collected over 12 weeks of sequential optimisation
across eight unknown functions.

The dataset was created by Burak Absoy, on behalf of himself as a learner on
the Imperial College London Executive Education ML/AI programme. The underlying
function evaluations are provided by Imperial College London as part of the
capstone challenge infrastructure.

---

## Composition

Each instance in the dataset represents a single query: an input vector
submitted to one of eight unknown black-box functions, paired with the
scalar output returned by that function.

**Function dimensionality and instance counts (as of Week 12):**

| Function | Dimensions | Input Range | Approx. Instances |
|----------|------------|-------------|-------------------|
| F1 | 2 | [0, 1) | ~22 |
| F2 | 2 | [0, 1) | ~22 |
| F3 | 3 | [0, 1) | ~22 |
| F4 | 4 | [0, 1) | ~22 |
| F5 | 4 | [0, 1) | ~22 |
| F6 | 5 | [0, 1) | ~22 |
| F7 | 6 | [0, 1) | ~22 |
| F8 | 8 | [0, 1) | ~22 |

Total: approximately 176 input-output pairs across all functions.
Input values are specified to six decimal places.
Output values are real-valued scalars with no fixed range.

**Missing data:** There are no missing values. Every submitted query received
a corresponding output from the portal. One accidental duplicate entry was
detected and removed after Week 8 using the `verifyAppend.py` tool.

**Confidential data:** The dataset contains no personal information,
private communications, or legally protected data. The function evaluations
are synthetic and provided solely for academic purposes.

**Output scale variation:**
- F1: near-zero throughout (~0)
- F2: range approximately [0.02, 0.66]
- F3: range approximately [-0.48, -0.009]
- F4: range approximately [-34.75, 0.48]
- F5: range approximately [1163, 8662]
- F6: range approximately [-3.06, -0.41]
- F7: range approximately [1.80, 3.08]
- F8: range approximately [9.24, 9.68]

---

## Collection Process

Queries were generated algorithmically using a Gaussian Process surrogate model
with RBF kernel, fitted on all previously observed data. Each week, one query
per function was submitted to the BBO portal, which returned the true function
output. Results were received via email and manually recorded into `inputs.txt`
and `outputs.txt` files, then appended to the `.npy` dataset files using
`appendToDataset.py`.

The dataset is not a sample of a larger dataset. Each query represents a
unique evaluation of the unknown function at a chosen input point. The
sampling strategy evolved across weeks:
- Weeks 1–3: exploration-focused (UCB, Variance, PI acquisition functions)
- Weeks 4–6: function-specific hybrid strategies introduced
- Weeks 7–12: tight local exploitation using Expected Improvement

All inputs were constrained to [0, 0.999999] to satisfy portal formatting
requirements. Values are specified to six decimal places.

The dataset was collected over a 12-week period from February to April 2026.

---

## Preprocessing/Cleaning/Labelling

**Transformations applied:**
- Z-score normalisation applied to outputs before GP fitting from Week 5 onwards.
  This was necessary to stabilise GP inference across functions with very different
  output scales (e.g. F5 outputs in the thousands vs F3 outputs near zero).
- Duplicate row detection and removal implemented after an accidental
  double-append was discovered in Week 8. The `verifyAppend.py` script
  now automatically removes duplicate rows before verification.

**Raw data preservation:** The original weekly submission files
(`inputs.txt` and `outputs.txt`) are preserved in their respective week
folders (`data/week_01/` through `data/week_12/`), providing a complete
audit trail of all submissions and outputs. The processed `.npy` files
are derived from these raw files.

---

## Uses

**Intended uses:**
- Benchmarking surrogate-based optimisation algorithms under low evaluation budgets
- Studying acquisition function behaviour across functions of varying dimensionality
- Analysing exploration-exploitation trade-offs in sequential decision-making
- Educational demonstration of Bayesian optimisation with Gaussian Process surrogates

**Potential impact on future uses:** The sampling distribution is heavily
biased toward regions identified as high-value in earlier weeks. Users
should be aware that the dataset does not represent uniform coverage of
the input space — particularly for higher-dimensional functions (6D, 8D),
large portions of the space are entirely unobserved. Any model trained on
this data will inherit this bias and may overestimate performance in the
sampled regions while having no information about the rest of the space.

**Tasks for which the dataset should not be used:**
- Inferring the analytical form of the underlying functions — the functions
  are unknown by design and the sparse observations do not support this
- Training deep learning models — the dataset is far too small
- Generalising conclusions to noisy or stochastic black-box settings —
  the functions appear deterministic (same input always returns same output)
- Any application involving real individuals, sensitive decisions, or
  consequential outcomes — this is a purely synthetic academic dataset

---

## Distribution

The dataset is publicly available in the GitHub repository:
`https://github.com/absoyak/imperial-ml-ai-capstone`

All data files are stored in the `capstone/data/` directory. The repository
is public for the duration of the Imperial College London programme.

The underlying function evaluations are provided by Imperial College London
and are not redistributable outside the programme context. No additional
copyright or IP licence applies to the query inputs and outputs generated
by the learner.

---

## Maintenance

The dataset is maintained by Burak Absoy as part of the Imperial College
London ML/AI Professional Certificate capstone project. Updates were made
weekly throughout the 12-week campaign. No further updates are expected
after programme completion.

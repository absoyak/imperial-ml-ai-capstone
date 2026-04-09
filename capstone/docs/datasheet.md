# Datasheet: BBO Capstone Project Dataset

## Motivation

This dataset was created to support a black-box optimisation (BBO) challenge
as part of the Imperial College London Machine Learning and AI Professional
Certificate programme. The dataset records all query inputs and corresponding
function evaluation outputs collected over 11 weeks of sequential optimisation
across eight unknown functions.

The dataset supports research into surrogate-based optimisation under strict
evaluation budgets, where the analytical form of each function is unknown and
information is revealed only through sequential query feedback.

---

## Composition

The dataset consists of two components:

**Seed data (initial observations):**
- 8 subdirectories, one per function: `data/function_1` through `data/function_8`
- Each contains `initial_inputs.npy` and `initial_outputs.npy`
- These files grow weekly as new observations are appended

**Weekly submission archive:**
- `data/week_01/` through `data/week_11/`: inputs and outputs for each weekly round
- Each folder contains `inputs.txt` and `outputs.txt` in Python list block format

**Function dimensionality:**

| Function | Dimensions | Input Range |
|----------|------------|-------------|
| F1 | 2 | [0, 1) |
| F2 | 2 | [0, 1) |
| F3 | 3 | [0, 1) |
| F4 | 4 | [0, 1) |
| F5 | 4 | [0, 1) |
| F6 | 5 | [0, 1) |
| F7 | 6 | [0, 1) |
| F8 | 8 | [0, 1) |

**Dataset size (as of Week 11):**
- Approximately 19–21 observations per function
- Total: approximately 160 input-output pairs across all functions
- Input values are specified to six decimal places
- Output values are real-valued scalars with no fixed range

**Output scale variation:**
- F1: near-zero throughout (~0)
- F2: range approximately [0.02, 0.66]
- F3: range approximately [-0.48, -0.01]
- F4: range approximately [-34.75, 0.48]
- F5: range approximately [1163, 4440]
- F6: range approximately [-3.06, -0.48]
- F7: range approximately [1.80, 3.08]
- F8: range approximately [9.24, 9.68]

---

## Collection Process

Queries were generated algorithmically using a Gaussian Process surrogate model
with RBF kernel, fitted on all previously observed data. Each week, one query
per function was submitted to the BBO portal, which returned the true function
output. Results were received via email and manually recorded.

The query strategy evolved across weeks:
- Weeks 1–3: exploration-focused (UCB, Variance, PI acquisition)
- Weeks 4–6: function-specific hybrid strategies introduced
- Weeks 7–11: tight local exploitation using Expected Improvement

All inputs were constrained to [0, 0.999999] to satisfy portal requirements.
Values are specified to six decimal places.

The dataset was collected over an 11-week period from February to April 2026.

---

## Preprocessing and Uses

**Transformations applied:**
- Y-normalisation (z-score) applied before GP fitting from Week 5 onwards
- Duplicate row detection and removal implemented after Week 8

**Intended uses:**
- Benchmarking surrogate-based optimisation algorithms under low evaluation budgets
- Studying acquisition function behaviour across functions of varying dimensionality
- Analysing exploration-exploitation trade-offs in sequential decision-making

**Inappropriate uses:**
- Direct inference about the analytical form of the underlying functions
- Training deep learning models (dataset is too small)
- Generalising conclusions to noisy black-box settings (functions appear deterministic)

---

## Distribution and Maintenance

The dataset is maintained in the GitHub repository:
`https://github.com/absoyak/imperial-ml-ai-capstone`

All data files are stored in the `capstone/data/` directory. The repository
is public for the duration of the programme.

The dataset is maintained by Burak Absoy as part of the Imperial College
London ML/AI Professional Certificate capstone project. No external licensing
applies; the underlying function evaluations are provided by Imperial College
London and are not redistributable outside the programme context.

**Known gaps and limitations:**
- Only one query per function per week — the dataset is extremely sparse
  relative to the dimensionality of higher-dimensional functions (6D, 8D)
- Queries are not uniformly distributed across the input space; later weeks
  are heavily concentrated around previously observed high-value regions
- Function 1 produced near-zero outputs across all 11 weeks, limiting the
  informativeness of those observations
- Function 4 showed high output variance with no consistent exploitable region

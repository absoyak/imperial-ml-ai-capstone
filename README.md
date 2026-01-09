# Imperial ML/AI Capstone — Work in Progress

This repository tracks my capstone work for the Imperial College London Professional Certificate in Machine Learning & Artificial Intelligence.

I’m using this space to document the problem definition, experiments, results, and decisions as the project evolves (code + notes + reproducibility).

---

## Project idea (draft)
**Topic:** Predicting performance in professional powerboat racing (F1H2O) using historical timing + race events + environmental measurements.

**Goal:** Build a model that can estimate expected lap time / session pace under given water and race conditions, and explore how much each factor contributes (conditions vs. team/boat vs. session context).

---

## Execution plan (how I’ll run this)
I’m running this capstone like a small real project: short weekly sprints, reproducible experiments, and a clear “done” definition.

- **Weekly rhythm:** 4 short sessions + 1 longer deep-work session
- **Milestones:** EDA → baseline + proper validation → features/models → tuning + error analysis → final clean write-up
- **Experiment tracking:** simple log + Git commits per change (no “mystery improvements”)
- **Validation:** split by event/time to avoid leakage (no random row splitting)
- **Risks I’m managing:** leakage, overfitting, endless tuning
- **What’s public:** code, notes, plots, write-up (no proprietary data)


## Why this is interesting
In live race operations, even small changes in water conditions or race context can affect lap times and safety decisions. A practical predictive model could help:
- Produce pre-session “expected pace” ranges
- Flag anomalies (unusually slow laps, potential issues)
- Support more data-driven planning and reporting

---

## Data (planned / partial)
Potential sources (subject to availability and cleaning):
- **Timing data:** lap times, pass times, sector/interval timing (if available)
- **Race events:** yellow flags, incidents, stoppages
- **Session metadata:** track/location, session type, date/time
- **Environmental data:** smart buoy measurements (e.g., water conditions)

> Notes: data will be anonymised/aggregated where needed. Any proprietary or sensitive data will not be uploaded.

---

## Approach (planned)
- **Baseline:** simple averages / rolling statistics by session + conditions
- **Features:** conditions + session context + recent lap history + event flags
- **Models to try:**
  - Linear / Ridge / Lasso (strong baseline + interpretability)
  - Tree-based (Random Forest / Gradient Boosting) for non-linear effects
  - Optional: time-series style features / sequence models if justified
- **Evaluation:**
  - MAE / RMSE for lap-time prediction
  - Train/test split by event or by time (avoid leakage)
  - Error analysis by session type and conditions

---

## Repository structure (will evolve)
```text
.
├── data/               # empty by default (or sample data only)
├── notebooks/          # exploration + experiments
├── src/                # reusable code
├── reports/            # write-ups / figures
└── README.md

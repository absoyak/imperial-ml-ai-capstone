## Data availability

The initial input–output observations used in this project are provided by the course organisers as NumPy (`.npy`) files via the Imperial College ML & AI programme platform.

These data files are not included in this repository, in line with the course submission guidelines and GitHub best practices for handling data.

To reproduce the experiments:
1. Download the initial data from the course platform.
2. Place each function’s data in a local directory structured as:
   `initial_data/function_X/initial_inputs.npy`
   and
   `initial_data/function_X/initial_outputs.npy`
3. Update local paths if required before running the optimisation scripts.

The optimisation code assumes no access to the internal structure of the objective functions and treats them strictly as black-box evaluations.

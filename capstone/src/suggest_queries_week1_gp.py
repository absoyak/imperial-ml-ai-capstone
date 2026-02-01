import os
import math
import numpy as np


# -----------------------------
# Configuration (edit if needed)
# -----------------------------

# You gave this exact path. Keep it as-is.
FUNCTION_1_FOLDER = r"C:\Users\absoy\OneDrive\Dev\ML\Capstone\initial_data\function_1"

# Number of random candidate points to evaluate per function.
# Increase for better suggestions (slower).
N_CANDIDATES = 50000

# GP noise term for numerical stability (small).
NOISE = 1e-8

# GP lengthscale for RBF kernel (can be tuned; this is a reasonable starting point).
LENGTHSCALE = 0.2

# Acquisition settings
UCB_KAPPA = 2.0   # Higher = more exploration
PI_XI = 1e-6      # Exploration term for PI


# -----------------------------
# Helpers
# -----------------------------

def rbf_kernel(x_a: np.ndarray, x_b: np.ndarray, lengthscale: float) -> np.ndarray:
    """
    RBF kernel: k(x, x') = exp(-||x-x'||^2 / (2*l^2))
    x_a: (n, d)
    x_b: (m, d)
    returns: (n, m)
    """
    # Squared Euclidean distance
    a2 = np.sum(x_a * x_a, axis=1).reshape(-1, 1)  # (n, 1)
    b2 = np.sum(x_b * x_b, axis=1).reshape(1, -1)  # (1, m)
    sq_dist = a2 + b2 - 2.0 * (x_a @ x_b.T)
    return np.exp(-sq_dist / (2.0 * (lengthscale ** 2)))


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF using erf; works for numpy arrays."""
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def gp_posterior(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                 lengthscale: float, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Basic GP regression (zero mean) with RBF kernel.
    Returns posterior mean and variance for x_test.
    """
    k_tt = rbf_kernel(x_train, x_train, lengthscale)  # (n, n)
    k_ts = rbf_kernel(x_train, x_test, lengthscale)   # (n, m)
    k_ss_diag = np.ones((x_test.shape[0],), dtype=np.float64)  # since k(x,x)=1 for RBF

    # Add noise on the diagonal for stability
    k_tt = k_tt + (noise * np.eye(k_tt.shape[0], dtype=np.float64))

    # Cholesky decomposition
    try:
        l = np.linalg.cholesky(k_tt)
    except np.linalg.LinAlgError:
        # Fallback: add more jitter if needed
        jitter = 1e-6
        k_tt = k_tt + (jitter * np.eye(k_tt.shape[0], dtype=np.float64))
        l = np.linalg.cholesky(k_tt)

    # Solve for alpha: K^-1 y via Cholesky
    y_train = y_train.reshape(-1, 1)
    v = np.linalg.solve(l, y_train)
    alpha = np.linalg.solve(l.T, v)  # (n, 1)

    # Posterior mean: k_ts^T alpha
    mu = (k_ts.T @ alpha).reshape(-1)  # (m,)

    # Posterior variance: k_ss - diag(k_ts^T K^-1 k_ts)
    # Compute: w = L^-1 k_ts
    w = np.linalg.solve(l, k_ts)  # (n, m)
    var = k_ss_diag - np.sum(w * w, axis=0)  # (m,)

    # Numerical safety
    var = np.maximum(var, 1e-12)

    return mu, var


def format_portal_input(x: np.ndarray) -> str:
    """
    Portal requires:
    - each value starts with 0.
    - exactly 6 decimals
    - hyphen separated, no spaces
    - values in [0.000000, 0.999999] (strictly < 1)
    """
    x = np.clip(x, 0.0, 0.999999)
    parts = [f"{v:.6f}" for v in x.tolist()]
    return "-".join(parts)


def infer_root_folder_from_function_1(function_1_folder: str) -> str:
    """
    If path ends with ...\\initial_data\\function_1, root is ...\\initial_data
    """
    return os.path.dirname(function_1_folder.rstrip("\\/"))


def load_initial_xy(function_folder: str) -> tuple[np.ndarray, np.ndarray]:
    x_path = os.path.join(function_folder, "initial_inputs.npy")
    y_path = os.path.join(function_folder, "initial_outputs.npy")
    x = np.load(x_path).astype(np.float64)
    y = np.load(y_path).astype(np.float64)
    if y.ndim != 1:
        y = y.reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X and Y row mismatch in {function_folder}: {x.shape[0]} vs {y.shape[0]}")
    return x, y


def suggest_next_point(x_train: np.ndarray, y_train: np.ndarray, acquisition: str,
                       n_candidates: int, lengthscale: float, noise: float,
                       ucb_kappa: float, pi_xi: float) -> tuple[np.ndarray, dict]:
    """
    Suggest one new point using random candidate evaluation.
    acquisition: 'ucb' | 'variance' | 'pi'
    """
    dim = x_train.shape[1]

    # Candidate points uniformly in [0, 0.999999)
    x_cand = np.random.rand(n_candidates, dim).astype(np.float64)
    x_cand = np.minimum(x_cand, 0.999999)

    mu, var = gp_posterior(x_train, y_train, x_cand, lengthscale, noise)
    sigma = np.sqrt(var)

    best_y = float(np.max(y_train))

    if acquisition.lower() == "ucb":
        score = mu + (ucb_kappa * sigma)
    elif acquisition.lower() == "variance":
        score = var
    elif acquisition.lower() == "pi":
        # Probability of improvement over current best
        # PI = Phi((mu - best - xi)/sigma)
        z = (mu - best_y - pi_xi) / (sigma + 1e-12)
        score = normal_cdf(z)
    else:
        raise ValueError("acquisition must be one of: 'ucb', 'variance', 'pi'")

    best_idx = int(np.argmax(score))
    x_next = x_cand[best_idx]

    debug = {
        "best_y_so_far": best_y,
        "mu_at_suggestion": float(mu[best_idx]),
        "sigma_at_suggestion": float(sigma[best_idx]),
        "acq_score": float(score[best_idx]),
        "acquisition": acquisition.lower(),
        "dim": dim,
        "n_candidates": n_candidates
    }

    return x_next, debug


# -----------------------------
# Main run
# -----------------------------

def main() -> None:
    root_folder = infer_root_folder_from_function_1(FUNCTION_1_FOLDER)

    # You can switch acquisition here to match your reflection:
    # 'variance' for strong exploration, 'ucb' balanced, 'pi' exploit-heavy.
    acquisition_to_use = "pi"   # change to:"ucb", "variance" or "pi"

    print(f"Root folder inferred: {root_folder}")
    print(f"Acquisition: {acquisition_to_use}")
    print("")

    for function_index in range(1, 9):
        function_folder = os.path.join(root_folder, f"function_{function_index}")

        x_train, y_train = load_initial_xy(function_folder)

        x_next, debug = suggest_next_point(
            x_train=x_train,
            y_train=y_train,
            acquisition=acquisition_to_use,
            n_candidates=N_CANDIDATES,
            lengthscale=LENGTHSCALE,
            noise=NOISE,
            ucb_kappa=UCB_KAPPA,
            pi_xi=PI_XI
        )

        portal_string = format_portal_input(x_next)

        print(f"Function {function_index} ({debug['dim']}D)")
        print(f"  Best Y so far     : {debug['best_y_so_far']}")
        print(f"  Suggestion (x)    : {portal_string}")
        print(f"  Pred mu / sigma   : {debug['mu_at_suggestion']} / {debug['sigma_at_suggestion']}")
        print("")

    print("Done. Copy the 'Suggestion (x)' lines into the portal fields (no spaces).")


if __name__ == "__main__":
    main()

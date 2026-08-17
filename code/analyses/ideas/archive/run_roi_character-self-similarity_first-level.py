from utils import *
from utils_fmri import *

import os
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import pdist

# -------------------------
# Config
# -------------------------
RESULTS_DIR = "../results/roi_analysis/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Use a NEW filename to avoid schema collisions with your previous memory-based output
OUT_CSV = os.path.join(RESULTS_DIR, "character-self-sim.csv")

ROIS  = ["HPC-L", "HPC-R", "PCC-L", "PCC-R"]
CHARS = CHARACTERS[:5]          # roles 1..5, consistent with utils.ROLE_TO_NAME

N_PERM      = 500               # permutations for circular-shift null
TIME_DEGREE = 1                 # 0 = none; 1 = remove linear drift; 2 = quadratic, etc.

# -------------------------
# Helpers
# -------------------------
def _stable_seed(sub_id, roi, base=0) -> int:
    s = f"{sub_id}|{roi}|{base}".encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    return int(h[:8], 16)

def _get_role_col(df):
    for c in ("character_role_num", "char_role_num", "character_id", "char_id"):
        if c in df.columns:
            return c
    raise ValueError("Missing character ID column (expected character_role_num/char_role_num/etc).")

def _fisher_mean_within_char(X):
    """
    X: (T,V) trials x voxels/features
    Returns Fisher-z mean of Pearson r across all within-character trial pairs.
    """
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        return np.nan

    d = pdist(X, metric="correlation")  # 1 - r; can contain NaNs if variance is 0
    if d.size == 0:
        return np.nan

    sim = 1.0 - d                       # r
    sim = np.clip(sim, -0.999999, 0.999999)
    z = np.arctanh(sim)

    # be robust to occasional NaNs from correlation distance
    return float(np.nanmean(z)) if np.any(np.isfinite(z)) else np.nan

def _split_by_char(betas, roles, n_chars=5):
    """
    betas: (T,V), roles: (T,) in 1..n_chars
    returns list Xc per character, and list n_trials per character
    """
    out = []
    ns  = []
    for c in range(1, n_chars + 1):
        idx = np.where(roles == c)[0]
        ns.append(int(idx.size))
        out.append(betas[idx])
    return out, ns

def _roll_rows(X, rng):
    """Circularly shift trial order by a non-zero k."""
    n = X.shape[0]
    if n <= 1:
        return X.copy()
    k = int(rng.integers(1, n))  # 1..n-1
    return np.roll(X, shift=k, axis=0)

def _perm_p_two_sided(null, obs):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if null.size == 0 or (not np.isfinite(obs)):
        return np.nan
    return float((1 + np.sum(np.abs(null) >= np.abs(obs))) / (1 + null.size))

def _z_from_null(obs, null, min_null=5):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if null.size < int(min_null) or (not np.isfinite(obs)):
        return np.nan, np.nan, np.nan
    mu = float(np.mean(null))
    sd = float(np.std(null, ddof=1))
    z  = float((obs - mu) / (sd + 1e-12))
    return mu, sd, z

def regress_out_temporal_trend(X, onsets, order=1):
    """
    Minimal polynomial detrend (feature-wise) using OLS with intercept.
    Returns residuals.
    """
    X = np.asarray(X, float)
    t = np.asarray(onsets, float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D; got {X.shape}")
    if t.ndim != 1 or t.shape[0] != X.shape[0]:
        raise ValueError(f"onsets must be (n_trials,); got {t.shape} vs {X.shape[0]}")

    if order <= 0:
        return X

    if not np.all(np.isfinite(t)):
        raise ValueError("Non-finite onsets encountered.")

    # center+scale for stability
    tc = t - np.mean(t)
    sd = np.std(tc, ddof=1)
    if sd < 1e-12:
        # no temporal variation
        return X - np.mean(X, axis=0)

    tc = tc / sd

    cols = [np.ones_like(tc)]
    for p in range(1, order + 1):
        cols.append(tc ** p)
    D = np.column_stack(cols)

    B, *_ = np.linalg.lstsq(D, X, rcond=None)
    X_hat = D @ B
    return X - X_hat

# -------------------------
# Output columns (fixed schema)
# -------------------------
BASE_COLS = [
    "sub_id", "roi",
    "time_degree",
    "n_perm_requested",
    "n_trials_total",
    "error", "error_msg",
]

CHAR_COLS = []
for c in CHARS:
    CHAR_COLS += [
        f"n_trials_{c}",
        f"sim_{c}",
        f"sim_null_mean_{c}",
        f"sim_null_sd_{c}",
        f"zsim_{c}",
        f"p_sim_{c}",
        f"n_perm_eff_sim_{c}",
    ]

ALL_COLS = BASE_COLS + CHAR_COLS

def _blank_out(sub_key, roi_key):
    out = {k: np.nan for k in ALL_COLS}
    out["sub_id"] = str(sub_key)
    out["roi"] = str(roi_key)
    out["time_degree"] = int(TIME_DEGREE)
    out["n_perm_requested"] = int(N_PERM)
    out["error"] = ""
    out["error_msg"] = ""
    return out

# Resume support
if os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 0:
    existing = pd.read_csv(OUT_CSV)
    processed = set(zip(existing["sub_id"].astype(str), existing["roi"].astype(str)))
else:
    processed = set()

# -------------------------
# Run
# -------------------------
subject_data = load_pickle("../analyses/lsa_decision_spm/subject_data_tavares-striatum.pkl")
all_incl_subs = incl_subs + incl_subs_tavares

for roi in tqdm(ROIS, desc="ROIs"):
    for sub_id in tqdm(all_incl_subs, desc=f"{roi}", leave=False):

        sub_key = str(sub_id)
        roi_key = str(roi)
        if (sub_key, roi_key) in processed:
            continue

        out = _blank_out(sub_key, roi_key)

        try:
            # Load and drop neutral trials (expects 60 or 63 in utils.drop_neutral_trials)
            betas = drop_neutral_trials(subject_data[sub_id]["roi_betas"][roi])   # (T,V)
            beh   = drop_neutral_trials(subject_data[sub_id]["behavior"])

            betas_arr = np.asarray(betas, float)
            role_col  = _get_role_col(beh)
            roles     = pd.to_numeric(beh[role_col], errors="coerce").fillna(-1).to_numpy(int)
            onsets    = pd.to_numeric(beh["onset"], errors="coerce").to_numpy(float)

            # Keep only roles 1..5, finite onsets, and finite beta rows
            keep = (
                (roles >= 1) & (roles <= len(CHARS)) &
                np.isfinite(onsets) &
                np.isfinite(betas_arr).all(axis=1)
            )
            betas_arr = betas_arr[keep]
            onsets    = onsets[keep]
            roles     = roles[keep]

            # Sort by onset so "time" is well-defined for circular shift
            ord_idx   = np.argsort(onsets)
            betas_arr = betas_arr[ord_idx]
            onsets    = onsets[ord_idx]
            roles     = roles[ord_idx]

            T = int(betas_arr.shape[0])
            out["n_trials_total"] = T

            if T < 2:
                raise ValueError("Too few trials after filtering to compute similarities.")

            # Optional detrend before computing similarity + nulls
            if int(TIME_DEGREE) > 0:
                betas_use = regress_out_temporal_trend(betas_arr, onsets, order=int(TIME_DEGREE))
            else:
                betas_use = betas_arr

            # Observed per-character self-similarity
            betas_by_char, n_trials_char = _split_by_char(betas_use, roles, n_chars=len(CHARS))
            sim_obs = np.full(len(CHARS), np.nan, float)
            for i, Xc in enumerate(betas_by_char):
                sim_obs[i] = _fisher_mean_within_char(Xc)

            # Null: circularly shift betas_use rows; keep roles fixed
            rng = np.random.default_rng(_stable_seed(sub_key, roi_key, base=12345))
            null_sim = [[] for _ in CHARS]

            for _ in range(int(N_PERM)):
                betas_perm = _roll_rows(betas_use, rng)
                betas_by_char_p, _ = _split_by_char(betas_perm, roles, n_chars=len(CHARS))

                for i, Xc in enumerate(betas_by_char_p):
                    s = _fisher_mean_within_char(Xc)
                    if np.isfinite(s):
                        null_sim[i].append(float(s))

            # Write per-character summaries
            for i, c in enumerate(CHARS):
                out[f"n_trials_{c}"] = int(n_trials_char[i])

                out[f"sim_{c}"] = float(sim_obs[i]) if np.isfinite(sim_obs[i]) else np.nan

                null_i = np.asarray(null_sim[i], float)
                out[f"n_perm_eff_sim_{c}"] = int(np.isfinite(null_i).sum())

                mu, sd, z = _z_from_null(sim_obs[i], null_i, min_null=5)
                out[f"sim_null_mean_{c}"] = mu
                out[f"sim_null_sd_{c}"]   = sd
                out[f"zsim_{c}"]          = z
                out[f"p_sim_{c}"]         = _perm_p_two_sided(null_i, sim_obs[i])

        except Exception as e:
            out["error"] = type(e).__name__
            out["error_msg"] = str(e)

        # Append to CSV
        write_header = (not os.path.exists(OUT_CSV)) or (os.path.getsize(OUT_CSV) == 0)
        pd.DataFrame([out], columns=ALL_COLS).to_csv(
            OUT_CSV, mode="a", header=write_header, index=False
        )

        processed.add((sub_key, roi_key))

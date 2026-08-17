from utils import *
from utils_fmri import *

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


# ------------------------------------------------------------
# Core stats
# ------------------------------------------------------------

def boundary_effect_from_mask(y_res, same_mask, min_per_group=3):
    """
    Compute mean(y_res | same) - mean(y_res | diff) using boolean same_mask.
    Invalid entries in y_res should already be NaN (excluded).
    """
    y_res = np.asarray(y_res, float)
    same_mask = np.asarray(same_mask, bool)

    valid = np.isfinite(y_res)
    if valid.sum() == 0:
        return np.nan, 0, 0

    ys = y_res[valid & same_mask]
    yd = y_res[valid & (~same_mask)]

    n_same = int(ys.size)
    n_diff = int(yd.size)
    if (n_same < min_per_group) or (n_diff < min_per_group):
        return np.nan, n_same, n_diff

    return float(np.mean(ys) - np.mean(yd)), n_same, n_diff


def circular_shift_null(y_res, same_mask, n_perm=2000, rng=None, min_per_group=3):
    """
    Legacy null: circularly shift same_mask and recompute boundary effect on y_res.
    Excludes shift=0 by sampling shifts in [1, L-1].
    """
    y_res = np.asarray(y_res, float)
    same_mask = np.asarray(same_mask, bool)

    L = int(same_mask.size)
    if L < 2:
        return np.full(int(n_perm), np.nan, float)

    if rng is None:
        rng = np.random.default_rng()

    null = np.full(int(n_perm), np.nan, float)
    for i in range(int(n_perm)):
        shift = int(rng.integers(1, L))  # 1..L-1
        sm = np.roll(same_mask, shift)
        null[i], _, _ = boundary_effect_from_mask(y_res, sm, min_per_group=min_per_group)

    return null


# ------------------------------------------------------------
# Stratification helpers (legacy option)
# ------------------------------------------------------------

def make_early_mid_late_strata(L, n_bins=3):
    """
    Index-based strata for transitions 0..L-1. Returns int array in {0..n_bins-1}.
    """
    L = int(L)
    if L <= 0:
        return np.array([], dtype=int)

    edges = np.linspace(0, L, int(n_bins) + 1)
    edges = np.round(edges).astype(int)
    edges[0] = 0
    edges[-1] = L
    edges = np.maximum.accumulate(edges)  # safety: non-decreasing

    strata = np.empty(L, dtype=int)
    for b in range(int(n_bins)):
        strata[edges[b]:edges[b + 1]] = b
    return strata


def make_early_mid_late_strata_from_time(onset, n_bins=3):
    """
    Time-based strata using transition midpoint time. onset length T -> strata length T-1.
    """
    onset = np.asarray(onset, float)
    if onset.size < 2:
        return np.array([], dtype=int)

    tmid = 0.5 * (onset[:-1] + onset[1:])
    if not np.isfinite(tmid).any():
        return np.zeros(tmid.size, dtype=int)

    qs = np.linspace(0, 1, int(n_bins) + 1)[1:-1]
    cuts = np.nanquantile(tmid, qs)
    return np.digitize(tmid, cuts, right=False).astype(int)


def stratified_perm_null(
    y_res,
    same_mask,
    strata,
    *,
    n_perm=2000,
    rng=None,
    min_per_group=3,
    permute_only_valid=True,
):
    """
    Permute same_mask within strata (e.g., early/mid/late).
    Preserves same/diff counts inside each stratum (within valid indices, if requested).
    """
    y_res = np.asarray(y_res, float)
    same_mask = np.asarray(same_mask, bool)
    strata = np.asarray(strata, int)

    L = int(same_mask.size)
    if L < 2 or strata.size != L:
        return np.full(int(n_perm), np.nan, float)

    if rng is None:
        rng = np.random.default_rng()

    null = np.full(int(n_perm), np.nan, float)
    valid = np.isfinite(y_res) if permute_only_valid else np.ones(L, dtype=bool)

    strata_vals = np.unique(strata[valid]) if valid.any() else np.unique(strata)
    idx_by_s = {s: np.where(valid & (strata == s))[0] for s in strata_vals}

    for i in range(int(n_perm)):
        sm = same_mask.copy()
        for _, idx in idx_by_s.items():
            if idx.size <= 1:
                continue
            sm[idx] = rng.permutation(sm[idx])

        null[i], _, _ = boundary_effect_from_mask(y_res, sm, min_per_group=min_per_group)

    return null


# ------------------------------------------------------------
# Freedman–Lane (circular-shift) null
# ------------------------------------------------------------

def _ols_fit(y, X):
    """
    Ordinary least squares via lstsq (robust to rank-deficiency).
    Returns (beta, y_hat, resid).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat
    return beta, y_hat, resid


def freedman_lane_circular_null(y, dt, same_mask, *, n_perm=2000, rng=None, min_per_group=3):
    """
    Freedman–Lane permutation test with a *circular-shift* operator.

    Full model:     y = b0 + b1*dt + b2*same + e
    Reduced model:  y = b0 + b1*dt + e

    FL procedure:
      1) Fit reduced -> y_hat0 + e0
      2) Circularly shift e0 (exclude shift=0)
      3) y* = y_hat0 + shifted(e0)
      4) Fit full to y*; record b2*

    Returns:
      null (n_perm,), obs_beta_same, n_same, n_diff
    """
    y = np.asarray(y, float)
    dt = np.asarray(dt, float)
    same_mask = np.asarray(same_mask, bool)

    if rng is None:
        rng = np.random.default_rng()

    valid = np.isfinite(y) & np.isfinite(dt)
    if valid.sum() < 2:
        return np.full(int(n_perm), np.nan, float), np.nan, 0, 0

    yv = y[valid]
    dtv = dt[valid]
    gv = same_mask[valid].astype(float)

    n_same = int(np.sum(gv == 1.0))
    n_diff = int(np.sum(gv == 0.0))
    if (n_same < min_per_group) or (n_diff < min_per_group):
        return np.full(int(n_perm), np.nan, float), np.nan, n_same, n_diff

    # observed statistic = coefficient on "same" in the full model
    X_full = np.column_stack([np.ones_like(dtv), dtv, gv])
    beta_full, _, _ = _ols_fit(yv, X_full)
    obs = float(beta_full[2])

    # reduced model fit
    X_red = np.column_stack([np.ones_like(dtv), dtv])
    _, yhat0, e0 = _ols_fit(yv, X_red)

    L = int(e0.size)
    if L < 2:
        return np.full(int(n_perm), np.nan, float), obs, n_same, n_diff

    null = np.full(int(n_perm), np.nan, float)
    for i in range(int(n_perm)):
        shift = int(rng.integers(1, L))  # 1..L-1 (exclude 0)
        y_star = yhat0 + np.roll(e0, shift)
        beta_star, _, _ = _ols_fit(y_star, X_full)
        null[i] = float(beta_star[2])

    return null, obs, n_same, n_diff


# ------------------------------------------------------------
# Main analysis function
# ------------------------------------------------------------

def character_boundary_effect_timecontrolled_perm(
    betas_by_trial,
    character_role_num,
    onset,
    *,
    n_perm=2000,
    rng=None,
    min_per_group=3,
    null_kind="circular",   # "circular"(Freedman–Lane) or "mask_circular" or "stratified"
    stratify_by="time",     # only used if null_kind=="stratified"
    n_strata=3,
    permute_only_valid=True,
):
    """
    Time-controlled character boundary effect on lag-1 Fisher-z(similarity).

    null_kind options:
      - "circular": Freedman–Lane circular-shift null (recommended)
      - "mask_circular": legacy null (circularly shift same_mask on residualized y)
      - "stratified": legacy null (permute same_mask within early/mid/late strata on residualized y)
    """
    X = np.asarray(betas_by_trial, float)
    chars = np.asarray(character_role_num)
    onset = np.asarray(onset, float)

    T = int(X.shape[0])
    if T < 2:
        return None
    if chars.size != T or onset.size != T:
        raise ValueError(f"Length mismatch: betas T={T}, chars={chars.size}, onset={onset.size}")

    # lag-1 similarity
    S = trial_correlation_matrix(X)
    lag1 = np.diagonal(S, offset=1)

    # Fisher-z
    y = np.arctanh(np.clip(lag1, -0.999999, 0.999999))

    # temporal distance
    dt = onset[1:] - onset[:-1]
    if np.nanmin(dt) < 0:
        dt = np.abs(dt)

    # same/diff mask + exclude invalid character transitions entirely
    c0 = chars[:-1]
    c1 = chars[1:]
    char_valid = np.isfinite(c0) & np.isfinite(c1)
    same_mask = (c0 == c1) & char_valid

    y_tc = np.asarray(y, float).copy()
    dt_tc = np.asarray(dt, float).copy()
    y_tc[~char_valid] = np.nan
    dt_tc[~char_valid] = np.nan

    out = {
        "boundary_effect_tc": np.nan,
        "n_same": 0,
        "n_diff": 0,
        "p_perm_right": np.nan,
        "p_perm_two": np.nan,
        "z_perm": np.nan,
        "null_mean": np.nan,
        "null_sd": np.nan,
        "null_kind": str(null_kind),
    }

    if rng is None:
        rng = np.random.default_rng()

    # --- Freedman–Lane circular-shift null ---
    if null_kind == "circular":
        null, obs, n_same, n_diff = freedman_lane_circular_null(
            y_tc, dt_tc, same_mask,
            n_perm=n_perm, rng=rng,
            min_per_group=min_per_group
        )
        out["boundary_effect_tc"] = float(obs) if np.isfinite(obs) else np.nan
        out["n_same"] = int(n_same)
        out["n_diff"] = int(n_diff)

    else:
        # Legacy: residualize then do label-based permutations
        y_res = regress_out(dt_tc, y_tc)  # residualize similarity on temporal distance
        y_res = np.asarray(y_res, float)

        obs, n_same, n_diff = boundary_effect_from_mask(y_res, same_mask, min_per_group=min_per_group)
        out["boundary_effect_tc"] = float(obs) if np.isfinite(obs) else np.nan
        out["n_same"] = int(n_same)
        out["n_diff"] = int(n_diff)

        if not np.isfinite(obs):
            return out

        if null_kind == "mask_circular":
            null = circular_shift_null(
                y_res, same_mask,
                n_perm=n_perm, rng=rng,
                min_per_group=min_per_group
            )

        elif null_kind == "stratified":
            L = same_mask.size
            if stratify_by == "time":
                strata = make_early_mid_late_strata_from_time(onset, n_bins=n_strata)
            else:
                strata = make_early_mid_late_strata(L, n_bins=n_strata)

            null = stratified_perm_null(
                y_res, same_mask, strata,
                n_perm=n_perm, rng=rng,
                min_per_group=min_per_group,
                permute_only_valid=permute_only_valid
            )
        else:
            raise ValueError(f"Unknown null_kind: {null_kind}")

    # --- post-processing ---
    obs = out["boundary_effect_tc"]
    if not np.isfinite(obs):
        return out

    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if null.size < max(50, int(n_perm * 0.2)):
        return out

    p_right = (np.sum(null >= obs) + 1.0) / (null.size + 1.0)
    p_two = (np.sum(np.abs(null) >= abs(obs)) + 1.0) / (null.size + 1.0)

    mu = float(np.mean(null))
    sd = float(np.std(null, ddof=1)) if null.size > 1 else np.nan
    z = (obs - mu) / (sd + 1e-12) if np.isfinite(sd) else np.nan

    out.update({
        "p_perm_right": float(p_right),
        "p_perm_two": float(p_two),
        "z_perm": float(z) if np.isfinite(z) else np.nan,
        "null_mean": float(mu),
        "null_sd": float(sd) if np.isfinite(sd) else np.nan,
    })
    return out


# ------------------------------------------------------------
# Run script (template style)
# ------------------------------------------------------------

results_dir = "../results/roi_analysis/"
os.makedirs(results_dir, exist_ok=True)
out_csv = os.path.join(results_dir, "character-boundary.csv")

if os.path.exists(out_csv):
    results_existing = pd.read_csv(out_csv)
    processed = set(zip(results_existing["sub_id"].astype(str), results_existing["roi"].astype(str)))
else:
    processed = set()

subject_data = load_pickle("../analyses/lsa_decision_spm/subject_data_tavares-striatum.pkl")

# If incl_subs isn't defined in your environment, fall back to keys in subject_data.
try:
    incl_subs
except NameError:
    incl_subs = sorted(list(subject_data.keys()))

# parameters
n_perm = 500
min_per_group = 3
null_kind = "circular"        # Freedman–Lane circular-shift null (recommended)
stratify_by = "time"          # only used if null_kind=="stratified"
n_strata = 3

# rois = ['HPC-L', 'HPC-R', 'PCC-L', 'PCC-R', 'DLPFC-L', 'DLPFC-R']
rois = list(subject_data[18002]['roi_betas'].keys())
all_incl_subs = incl_subs + incl_subs_tavares
for roi in tqdm(rois, desc="ROIs"):
    for sub_id in tqdm(all_incl_subs, desc=f"{roi}", leave=False):
        sub_key = str(sub_id)
        roi_key = str(roi)

        if (sub_key, roi_key) in processed:
            continue

        row = {
            "sub_id": sub_key,
            "roi": roi_key,
            "obs": np.nan,
            "null_mean": np.nan,
            "null_sd": np.nan,
            "z": np.nan,
            "n_same": np.nan,
            "n_diff": np.nan,
            "null_kind": str(null_kind),
        }

        try:
            betas = subject_data[sub_id]["roi_betas"][roi]
            beh = subject_data[sub_id]["behavior"]
            chars = beh["character_role_num"].to_numpy()
            onsets = beh["onset"].to_numpy()

            # deterministic per-(sub,roi) RNG
            seed = stable_seed("boundary", sub_key, roi_key, n_perm, null_kind, stratify_by, n_strata)
            rng = np.random.default_rng(seed)

            b = character_boundary_effect_timecontrolled_perm(
                betas, chars, onsets,
                n_perm=n_perm,
                rng=rng,
                min_per_group=min_per_group,
                null_kind=null_kind,
                stratify_by=stratify_by,
                n_strata=n_strata,
            )

            row.update({
                "obs": b.get("boundary_effect_tc", np.nan),
                "null_mean": b.get("null_mean", np.nan),
                "null_sd": b.get("null_sd", np.nan),
                "z": b.get("z_perm", np.nan),  # NOTE: fixed (was b.get("z", ...))
                "n_same": b.get("n_same", np.nan),
                "n_diff": b.get("n_diff", np.nan),
                "null_kind": b.get("null_kind", str(null_kind)),
            })

        except Exception:
            # leave NaNs; still write row so we don't retry forever
            pass

        write_header = (not os.path.exists(out_csv)) or (os.path.getsize(out_csv) == 0)
        pd.DataFrame([row]).to_csv(
            out_csv,
            mode="a",
            header=write_header,
            index=False,
        )

        processed.add((sub_key, roi_key))

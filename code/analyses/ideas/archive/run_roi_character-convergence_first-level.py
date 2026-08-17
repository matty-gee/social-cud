from utils import *
import hashlib


# -----------------------
# Small helpers
# -----------------------

def stable_seed(*parts) -> int:
    """
    Deterministic 32-bit seed from arbitrary inputs.
    """
    s = "||".join(map(str, parts))
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit-ish

def _existing_cols(csv_path):
    """
    Return existing CSV columns (stable schema) or None if file doesn't exist/empty.
    """
    if (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0):
        return None
    return list(pd.read_csv(csv_path, nrows=0).columns)

def _zscore_1d(x):
    x = np.asarray(x, float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if (not np.isfinite(sd)) or (sd <= 0):
        return np.zeros_like(x, dtype=float)
    return (x - mu) / (sd + 1e-12)

def _safe_row_corr(a, b, min_feats=3):
    """
    Pearson correlation between two 1D vectors with NaN/Inf handling.
    Uses only features that are finite in both.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < int(min_feats):
        return np.nan

    aa = a[ok]
    bb = b[ok]

    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))

    denom = float(np.sqrt((aa @ aa) * (bb @ bb)))
    if denom <= 0 or (not np.isfinite(denom)):
        return np.nan
    return float((aa @ bb) / denom)

def _lag1_similarity_fisherz(X):
    """
    Compute lag-1 correlation between consecutive rows of X,
    then Fisher-z transform.
    Returns y (length T-1) with NaNs where invalid.
    """
    X = np.asarray(X, float)
    T = int(X.shape[0])
    if T < 2:
        return np.array([], dtype=float)

    r = np.full(T - 1, np.nan, float)
    for t in range(T - 1):
        r[t] = _safe_row_corr(X[t], X[t + 1])

    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def _poly_cols(v, order):
    v = np.asarray(v, float)
    cols = []
    for p in range(1, int(order) + 1):
        cols.append(v ** p)
    return cols

# -----------------------
# Freedman–Lane + circular shift for within-character convergence
# -----------------------

def fit_withinchar_convergence_freedman_lane(
    betas_by_trial,
    character_role_num,
    character_decision_num,
    onset,
    *,
    n_perm=500,
    seed=0,
    min_pairs=3,
    time_order=2,
    include_dt=True,
    include_tmid=True,
    zscore_y=True,
    zscore_x=True,
    return_null=False,
):
    """
    Within-character convergence test with time-control nuisance model and
    Freedman–Lane + circular-shift permutations.

    Model (within-character adjacent pairs only):
        y ~ (time nuisance terms...) + x + intercept

    where:
      - y is Fisher-z(lag1 corr between consecutive trials' betas)
      - x is decision number (character_decision_num) of the first element in each adjacent pair
      - time nuisances include dt and/or tmid with polynomial expansion up to time_order

    Freedman–Lane (circular shift):
      1) Fit nuisance-only -> yhat_Z, resid
      2) For each permutation: resid_perm = roll(resid, k), y* = yhat_Z + resid_perm
      3) Fit full model to y* -> beta_x
      4) Return observed beta_x, null mean/sd, z, and permutation p-values.
    """
    X = np.asarray(betas_by_trial, float)
    chars = np.asarray(character_role_num)
    decs  = np.asarray(character_decision_num, float)
    onset = np.asarray(onset, float)

    T = int(X.shape[0])
    out = {
        "n_pairs": 0,
        "n_perm": int(n_perm),
        "beta_decision": np.nan,
        "null_mean_decision": np.nan,
        "null_sd_decision": np.nan,
        "z_decision": np.nan,
        "p_perm_right": np.nan,
        "p_perm_two": np.nan,
        "time_order": int(time_order),
        "include_dt": bool(include_dt),
        "include_tmid": bool(include_tmid),
        "null_kind": "freedman_lane_circular_shift",
    }

    if T < 2:
        return out
    if chars.size != T or decs.size != T or onset.size != T:
        raise ValueError(f"Length mismatch: betas T={T}, chars={chars.size}, decs={decs.size}, onset={onset.size}")

    # outcome: lag-1 Fisher-z similarity
    y = _lag1_similarity_fisherz(X)  # length T-1

    # within-character adjacent transitions
    same_mask = (chars[:-1] == chars[1:]) & np.isfinite(chars[:-1]) & np.isfinite(chars[1:])
    x = decs[:-1]  # decision # for the first element of each adjacent pair

    # time nuisances (length T-1)
    dt = onset[1:] - onset[:-1]
    if np.isfinite(dt).any() and (np.nanmin(dt) < 0):
        dt = np.abs(dt)
    tmid = 0.5 * (onset[1:] + onset[:-1])

    # Build validity mask (within-character + finite y/x + finite nuisances used)
    valid = same_mask & np.isfinite(y) & np.isfinite(x)
    if include_dt:
        valid &= np.isfinite(dt)
    if include_tmid:
        valid &= np.isfinite(tmid)

    n = int(valid.sum())
    out["n_pairs"] = n
    if n < int(min_pairs):
        return out

    # Extract valid rows in chronological order
    yv = y[valid].astype(float)
    xv = x[valid].astype(float)

    # Optional z-scoring for stability/standardized beta
    if zscore_y:
        yv = _zscore_1d(yv)
    if zscore_x:
        xv = _zscore_1d(xv)

    Z_cols = []
    if include_dt:
        Z_cols.extend(_poly_cols(dt[valid], time_order))
    if include_tmid:
        # Center tmid before polynomial expansion to reduce collinearity
        tmid_v = tmid[valid].astype(float)
        tmid_v = tmid_v - float(np.mean(tmid_v))
        Z_cols.extend(_poly_cols(tmid_v, time_order))

    # z-score nuisance cols (robustness)
    Z_cols = [ _zscore_1d(col) for col in Z_cols ]

    ones = np.ones(n, float)

    # Full model: intercept + nuisance + x
    XZ = np.column_stack([ones] + Z_cols + [xv])
    # Nuisance-only: intercept + nuisance
    ZM = np.column_stack([ones] + Z_cols) if len(Z_cols) > 0 else ones[:, None]

    # Observed beta_x
    beta_full = np.linalg.lstsq(XZ, yv, rcond=None)[0]
    beta_x_obs = float(beta_full[-1])
    out["beta_decision"] = beta_x_obs

    # Fit nuisance-only for Freedman–Lane
    beta_nuis = np.linalg.lstsq(ZM, yv, rcond=None)[0]
    yhat_Z = ZM @ beta_nuis
    resid = yv - yhat_Z

    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, max(n, 2), size=int(n_perm))  # exclude 0
    null_betas = np.full(int(n_perm), np.nan, float)

    for i, k in enumerate(shifts):
        resid_perm = np.roll(resid, int(k))
        y_star = yhat_Z + resid_perm
        beta_star = np.linalg.lstsq(XZ, y_star, rcond=None)[0]
        null_betas[i] = float(beta_star[-1])

    null = null_betas[np.isfinite(null_betas)]
    if null.size < max(50, int(n_perm * 0.2)):
        # too few valid permutations -> return observed only
        if return_null:
            out["null_betas_decision"] = null_betas
        return out

    mu = float(np.mean(null))
    sd = float(np.std(null, ddof=1)) if null.size > 1 else np.nan
    z  = (beta_x_obs - mu) / (sd + 1e-12) if np.isfinite(sd) else np.nan

    # Right-tail: expect positive convergence beta
    p_right = (np.sum(null >= beta_x_obs) + 1.0) / (null.size + 1.0)
    p_two   = (np.sum(np.abs(null) >= abs(beta_x_obs)) + 1.0) / (null.size + 1.0)

    out.update({
        "null_mean_decision": mu,
        "null_sd_decision": sd if np.isfinite(sd) else np.nan,
        "z_decision": float(z) if np.isfinite(z) else np.nan,
        "p_perm_right": float(p_right),
        "p_perm_two": float(p_two),
    })

    if return_null:
        out["null_betas_decision"] = null_betas

    return out


# -----------------------
# Run like your RSA loop
# -----------------------
results_dir = "../results/roi_analysis/"
os.makedirs(results_dir, exist_ok=True)
out_csv = os.path.join(results_dir, "character-convergence.csv")

if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
    results_existing = pd.read_csv(out_csv)
    processed = set(zip(results_existing["sub_id"].astype(str), results_existing["roi"].astype(str)))
else:
    processed = set()

subject_data = load_pickle("../analyses/lsa_decision_spm/subject_data_tavares-striatum.pkl")
rois = ["HPC-L", "HPC-R", "PCC-L", "PCC-R", "DLPFC-L", "DLPFC-R"]

# parameters
n_perm = 500
min_pairs = 3
time_order = 1   # time nuisance polynomial order (dt^1..dt^order and tmid^1..tmid^order)
all_incl_subs = incl_subs + incl_subs_tavares
rois = list(subject_data[18002]['roi_betas'].keys())

for roi in tqdm(rois, desc="ROIs"):
    for sub_id in tqdm(all_incl_subs, desc=f"{roi}", leave=False):
        sub_key = str(sub_id)
        roi_key = str(roi)
        if (sub_key, roi_key) in processed:
            continue

        try:
            betas    = subject_data[sub_id]["roi_betas"][roi]
            behavior = subject_data[sub_id]["behavior"]

            chars  = behavior["character_role_num"].to_numpy()
            decs   = behavior["character_decision_num"].to_numpy(float)
            onset  = behavior["onset"].to_numpy(float)

            seed = stable_seed(
                "withinchar_FL_circshift",
                sub_key, roi_key,
                n_perm, min_pairs, time_order
            )

            res = fit_withinchar_convergence_freedman_lane(
                betas, chars, decs, onset,
                n_perm=n_perm,
                seed=seed,
                min_pairs=min_pairs,
                time_order=time_order,
                include_dt=True,
                include_tmid=True,   # this is the "control slow drift over time" piece
                zscore_y=True,
                zscore_x=True,
                return_null=False
            )

            row = {"sub_id": sub_key, "roi": roi_key, **res}

        except Exception:
            row = {"sub_id": sub_key, "roi": roi_key}

        # Keep CSV schema stable once it exists
        cols = _existing_cols(out_csv)
        if cols is None:
            cols = list(row.keys())
        else:
            for c in cols:
                row.setdefault(c, np.nan)

        write_header = (not os.path.exists(out_csv)) or (os.path.getsize(out_csv) == 0)
        pd.DataFrame([row], columns=cols).to_csv(
            out_csv,
            mode="a",
            header=write_header,
            index=False,
        )

        processed.add((sub_key, roi_key))

from utils import *
from utils_fmri import *

import os
import hashlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import subspace_angles

results_dir = "../results/roi_analysis/"
os.makedirs(results_dir, exist_ok=True)
out_csv = os.path.join(results_dir, "subspace-angles.csv")

# ---------------------------------------------------------
# Resume logic
# ---------------------------------------------------------

if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
    results_existing = pd.read_csv(out_csv)
    processed = set(zip(results_existing["sub_id"].astype(str),
                        results_existing["roi"].astype(str)))
else:
    processed = set()

def _existing_cols(path):
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        return None
    return list(pd.read_csv(path, nrows=0).columns)

def _stable_seed(sub_id, roi_key, base=0) -> int:
    s = f"{sub_id}|{roi_key}|{base}".encode("utf-8")
    h = hashlib.md5(s).hexdigest()
    return int(h[:8], 16)  # 32-bit-ish

# ---------------------------------------------------------
# Core: coordinate subspace and principal angles
# ---------------------------------------------------------

def coord_subspace(betas_c, coords_c, k=2, ridge=1e-6, min_trials=5):
    """
    Estimate a voxel-space subspace capturing coordinate-related signal for ONE character.

    betas_c : (T, V)
    coords_c: (T, 2)
    Returns Q: (V, k_eff) orthonormal basis, or None if insufficient/invalid data.
    """
    Y = np.asarray(betas_c, float)
    X = np.asarray(coords_c, float)

    if Y.ndim != 2 or X.ndim != 2 or X.shape[1] != 2:
        return None
    T, V = Y.shape
    if T < min_trials or V == 0:
        return None

    # Drop rows with NaNs/Infs in either Y or X
    ok = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    Y = Y[ok]
    X = X[ok]
    if Y.shape[0] < min_trials:
        return None

    # Z-score coords (guard against zero variance)
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True)
    sd = np.maximum(sd, 1e-8)
    Xz = (X - mu) / sd

    # k cannot exceed 2 predictors nor voxel count
    k_eff = int(min(k, 2, V))
    if k_eff < 1:
        return None

    # Ridge multivariate regression: B = (X'X + λI)^-1 X'Y  -> (2, V)
    XtX = Xz.T @ Xz
    try:
        B = np.linalg.solve(XtX + ridge * np.eye(2), Xz.T @ Y)  # (2, V)
    except np.linalg.LinAlgError:
        return None

    if not np.isfinite(B).all():
        return None

    # SVD to get voxel directions
    try:
        _, _, VT = np.linalg.svd(B, full_matrices=False)
    except Exception:
        return None

    basis = VT[:k_eff].T  # (V, k_eff)
    if basis.size == 0 or not np.isfinite(basis).all():
        return None

    # Orthonormalize
    try:
        Q, _ = np.linalg.qr(basis, mode="reduced")
    except np.linalg.LinAlgError:
        return None

    if Q.shape[1] < k_eff:
        return None

    return Q[:, :k_eff]

def principal_angles_list(betas_by_char, coords_by_char, k=2, min_trials=5):
    """
    betas_by_char : list of (T_i, V)
    coords_by_char: list of (T_i, 2)

    Returns:
      angles_rad: dict {(i,j): angles (k,)} in radians
      bases: list of Q bases (or None) per character
    """
    assert len(betas_by_char) == len(coords_by_char)
    nC = len(betas_by_char)

    bases = []
    for i in range(nC):
        Q = coord_subspace(betas_by_char[i], coords_by_char[i], k=k, min_trials=min_trials)
        bases.append(Q)

    angles_rad = {}
    for i in range(nC):
        if bases[i] is None:
            continue
        for j in range(i + 1, nC):
            if bases[j] is None:
                continue
            ang = subspace_angles(bases[i], bases[j])  # length k, radians
            angles_rad[(i, j)] = ang

    return angles_rad, bases

# ---------------------------------------------------------
# Null: FREEDMAN–LANE + GLOBAL circular shift (before character splitting)
# ---------------------------------------------------------

def freedman_lane_circular_shift_time_series(X, onsets, rng, *, degree=1, min_shift=1, eps=1e-8):
    """
    Freedman–Lane-style restricted permutation for a trialwise time series X:

      1) Fit reduced model X ~ Z(time)  (Z = intercept + poly(time))
      2) Residuals R = X - ZB
      3) Circularly shift R in *temporal order* (preserves autocorrelation up to wrap)
      4) Recompose: X_perm = ZB + R_shift

    Parameters
    ----------
    X : (T, D) or (T,) array
        Trialwise series to permute (here: coords with D=2).
    onsets : (T,) array
        Trial onsets defining temporal order.
    degree : int
        Polynomial degree of time nuisance preserved in absolute time.
        degree=0 reduces to intercept-only FL (equivalent to global circular shift).
    min_shift : int
        Minimum shift size; min_shift=1 avoids the identity permutation.
    """
    
    def _poly_time_design(onsets, degree=1, eps=1e-8):
        """
        Build nuisance design: intercept + poly(zscored_time) up to 'degree'.
        """
        t = np.asarray(onsets, float)
        if not np.isfinite(t).all():
            # fall back to index time
            t = np.arange(len(t), dtype=float)

        tn = (t - np.mean(t)) / (np.std(t) + eps)

        X = [np.ones(len(tn), float)]
        for d in range(1, int(degree) + 1):
            X.append(tn ** d)
        return np.column_stack(X)  # (T, p)

    
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]

    T, D = X.shape
    if T <= 1:
        return X.squeeze().copy()

    # nuisance design and reduced-model fit
    Z = _poly_time_design(onsets, degree=int(degree), eps=eps)  # (T, p)
    B = np.linalg.lstsq(Z, X, rcond=None)[0]                   # (p, D)
    fit = Z @ B                                                # (T, D)
    R = X - fit                                                # (T, D)

    # shift residuals in temporal order
    t = np.asarray(onsets, float)
    if not np.isfinite(t).all():
        t = np.arange(T, dtype=float)

    order = np.argsort(t, kind="mergesort")
    R_ord = R[order]

    lo = int(max(1, min_shift))
    if T <= lo:
        lo = 1

    if T == 2:
        s = 1
    else:
        s = int(rng.integers(lo, T))  # {lo, ..., T-1}

    R_shift = np.roll(R_ord, shift=s, axis=0)

    R_perm = R.copy()
    R_perm[order] = R_shift

    X_perm = fit + R_perm
    return X_perm.squeeze()

def _split_by_character_sorted(betas, coords, onsets, roles, n_chars=5):
    """
    Split trials by character role into lists. Sorting within character is optional for estimation
    (regression is order-invariant), but kept for reproducibility / consistent temporal indexing.
    """
    betas = np.asarray(betas, float)
    coords = np.asarray(coords, float)
    onsets = np.asarray(onsets, float) if onsets is not None else None
    roles = np.asarray(roles, int)

    out_b, out_c = [], []
    for c in range(1, int(n_chars) + 1):
        idx = np.where(roles == c)[0]
        if idx.size == 0:
            out_b.append(np.empty((0, betas.shape[1]), float))
            out_c.append(np.empty((0, 2), float))
            continue
        if onsets is not None and np.isfinite(onsets[idx]).all():
            idx = idx[np.argsort(onsets[idx])]
        out_b.append(betas[idx])
        out_c.append(coords[idx])
    return out_b, out_c

# ---------------------------------------------------------
# Per-subject test
# ---------------------------------------------------------

def subject_principal_angle_test(
    sd,
    roi="HPC-L",
    *,
    k=2,
    n_perm=500,
    seed=0,
    time_degree=1,          # detrend betas (optional)
    null_time_degree=None,  # preserved time nuisance in FL perm of coords (defaults to time_degree)
    min_trials=5,
    include_neutral=False,
):
    """
    Per-subject permutation test for cross-character overlap of coordinate-related subspaces.

    Summary statistic (per subject):
      T_true = median over character-pairs of the FIRST principal angle (degrees).
      Smaller angle => more overlap / more "shared map".

    Null (FL + global circular shift BEFORE character splitting):
      - Fit coords ~ poly(time) (reduced model), shift residuals in time order, add back fit.
      - Split permuted coords by ORIGINAL character labels (roles) and recompute angles.

    This preserves:
      - the character-specific trial partition of betas (which trials belong to which character)
      - the global temporal autocorrelation structure of coords residuals (up to wrap-around)
      - the chosen low-frequency time trend in coords in absolute time (degree>=1)

    It breaks:
      - the trial-locked mapping between coords and neural patterns within each character.
    """
    rng = np.random.default_rng(seed)

    betas = sd["roi_betas"][roi]
    behavior = sd["behavior"]

    # Optional: drop neutral trials (63 -> 60)
    if (len(behavior) == 63) and (not include_neutral):
        behavior = drop_neutral_trials(behavior)
        if hasattr(betas, "shape") and betas.shape[0] == 63:
            betas = drop_neutral_trials(betas)

    betas_arr = betas.to_numpy(float) if isinstance(betas, pd.DataFrame) else np.asarray(betas, float)

    # Onsets define temporal order
    if "onset" in behavior.columns:
        onsets = pd.to_numeric(behavior["onset"], errors="coerce").to_numpy(float)
        if not np.isfinite(onsets).all():
            onsets = np.arange(len(behavior), dtype=float)
    else:
        onsets = np.arange(len(behavior), dtype=float)

    # Character roles
    role_col = None
    for c in ("character_role_num", "char_role_num", "character_id", "char_id"):
        if c in behavior.columns:
            role_col = c
            break
    if role_col is None:
        raise ValueError("Behavior is missing a character ID column (expected character_role_num/char_role_num).")

    roles = pd.to_numeric(behavior[role_col], errors="coerce").fillna(-1).to_numpy(int)

    # Coordinates
    coords = behavior[["affil_coord", "power_coord"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    # Keep only real character trials + finite betas/coords
    n_chars = 6 if include_neutral else 5
    keep_role = (roles >= 1) & (roles <= n_chars)
    keep = keep_role & np.isfinite(coords).all(axis=1) & np.isfinite(betas_arr).all(axis=1)

    betas_arr = betas_arr[keep]
    coords = coords[keep]
    onsets = onsets[keep]
    roles = roles[keep]

    n_trials = int(len(roles))
    if n_trials < 10:
        return None

    # Optional temporal detrending of betas across kept trials
    if int(time_degree) > 0:
        betas_arr = regress_out_temporal_trend(
            betas_arr,
            onsets,
            order=int(time_degree),
            include_intercept=True,
            keep_mean=False,
            drop_nonfinite=False,
        )

    # True statistic
    betas_by_char, coords_by_char = _split_by_character_sorted(
        betas_arr, coords, onsets, roles, n_chars=n_chars
    )

    angles_rad, _ = principal_angles_list(betas_by_char, coords_by_char, k=k, min_trials=min_trials)
    if len(angles_rad) == 0:
        return None

    angles_deg = np.array([np.degrees(a[0]) for a in angles_rad.values()], float)
    angles_deg = angles_deg[np.isfinite(angles_deg)]
    if angles_deg.size == 0:
        return None

    T_true = float(np.median(angles_deg))

    # Null distribution: FL global shift of coords BEFORE splitting
    if null_time_degree is None:
        null_time_degree = int(time_degree)

    T_null = []
    for _ in range(int(n_perm)):
        coords_perm_trialwise = freedman_lane_circular_shift_time_series(
            coords,
            onsets,
            rng,
            degree=int(null_time_degree),
            min_shift=1,
        )

        bpc, cpc = _split_by_character_sorted(
            betas_arr, coords_perm_trialwise, onsets, roles, n_chars=n_chars
        )

        ang_p, _ = principal_angles_list(bpc, cpc, k=k, min_trials=min_trials)
        if len(ang_p) == 0:
            continue

        ang0 = np.array([np.degrees(a[0]) for a in ang_p.values()], float)
        ang0 = ang0[np.isfinite(ang0)]
        if ang0.size == 0:
            continue

        T_null.append(float(np.median(ang0)))

    T_null = np.asarray(T_null, float)
    T_null = T_null[np.isfinite(T_null)]
    if T_null.size == 0:
        return None

    null_mean = float(np.mean(T_null))
    null_sd = float(np.std(T_null, ddof=1))

    # One-sided p-value: smaller angle => more overlap
    p_perm = float((1 + np.sum(T_null <= T_true)) / (1 + T_null.size))

    # z-score: positive => more overlap than null
    z_perm = float((null_mean - T_true) / (null_sd + 1e-12))

    return dict(
        n_trials_used=n_trials,
        n_pairs=int(angles_deg.size),
        n_perm_eff=int(T_null.size),
        T_true=T_true,
        null_mean=null_mean,
        null_sd=null_sd,
        z_perm=z_perm,
        p_perm=p_perm,
        null_time_degree=int(null_time_degree),
    )

# -------------------------------
# Load data + run ROI analysis
# -------------------------------

subject_data = load_pickle("../analyses/lsa_decision_spm/subject_data_tavares-striatum.pkl")
rois = list(subject_data[next(iter(subject_data))]["roi_betas"].keys())
all_incl_subs = incl_subs + incl_subs_tavares

K = 2

N_PERM = 500
MIN_TRIALS_PER_CHAR = 5
TIME_DEGREE = 1  # set to 0 for intercept-only (pure global circular shift)

for roi in tqdm(rois, desc="ROIs"):
    for sub_id in tqdm(all_incl_subs, desc=f"{roi}", leave=False):
        sub_key = str(sub_id)
        roi_key = str(roi)
        if (sub_key, roi_key) in processed:
            continue

        try:
            sd = subject_data[sub_id]
            seed = _stable_seed(sub_key, roi_key, base=12345)
            out = subject_principal_angle_test(
                sd,
                roi=roi,
                k=K,
                n_perm=N_PERM,
                seed=seed,
                time_degree=TIME_DEGREE,
                null_time_degree=TIME_DEGREE,
                min_trials=MIN_TRIALS_PER_CHAR,
                include_neutral=False,
            )

            if out is None:
                raise ValueError("No valid character-pair angles (insufficient trials or invalid subspaces).")

            row = {
                "sub_id": sub_key,
                "roi": roi_key,
                "k": int(K),
                "time_degree": int(TIME_DEGREE),
                "n_perm_requested": int(N_PERM),
                **out,
            }

        except Exception as e:
            row = {
                "sub_id": sub_key,
                "roi": roi_key,
                "k": int(K),
                "time_degree": int(TIME_DEGREE),
                "n_perm_requested": int(N_PERM),
                "error": type(e).__name__,
                "error_msg": str(e),
            }

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

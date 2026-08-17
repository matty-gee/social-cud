from utils import *  
from utils_fmri import regress_out_temporal_trend
from scipy.stats import zscore



# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

results_dir = "../results/roi_analysis/"
os.makedirs(results_dir, exist_ok=True)

out_csv = os.path.join(results_dir, "axis-angles.csv")
if os.path.exists(out_csv):
    results_existing = pd.read_csv(out_csv)
    processed = set(zip(results_existing["sub_id"], results_existing["roi"]))
else:
    processed = set()


# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------


subject_data = load_pickle("../analyses/lsa_decision_spm/subject_data_tavares-striatum.pkl")
all_incl_subs = incl_subs + incl_subs_tavares
rois = list(subject_data[all_incl_subs[0]]["roi_betas"].keys())


# ------------------------------------------------------------
# Hyperparameter grid + CV settings
# ------------------------------------------------------------

lam_grid = np.array([0.1, 1.0, 10.0, 100.0], dtype=float)  # small grid; adjust as needed
n_folds = 5
shuffle_folds = True  # set False if you want contiguous folds
seed = 0

# Permutations (Freedman–Lane circular shift)
n_perm = 200


# ------------------------------------------------------------
# Helpers: ridge / design preparation
# ------------------------------------------------------------
def _filter_rows(Y, aff, pow_):
    """
    Apply a single validity mask once (finite aff/pow + finite Y rows).
    Returns Yf, afff, powf, and n_trials.
    """
    Y = np.asarray(Y, float)
    aff = np.asarray(aff, float)
    pow_ = np.asarray(pow_, float)

    m = np.isfinite(aff) & np.isfinite(pow_) & np.isfinite(Y).all(axis=1)
    Yf = Y[m]
    afff = aff[m]
    powf = pow_[m]
    return Yf, afff, powf, int(Yf.shape[0])


def _ridge_fit_B(X, Y, lam):
    """
    Ridge regression B = argmin ||Y - X B||^2 + lam||B_nonint||^2
    Intercept is NOT penalized.
    X: (T,3) [1, a, p], Y: (T,V) -> B: (3,V)
    """
    P = np.diag([0.0, float(lam), float(lam)])
    return np.linalg.solve(X.T @ X + P, X.T @ Y)


def _cos_angle(u, v):
    den = float(np.linalg.norm(u) * np.linalg.norm(v))
    if not np.isfinite(den) or den <= 0:
        return np.nan, np.nan
    c = float(np.dot(u, v) / den)
    a = float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return c, a


def _make_folds(T, n_splits=5, seed=0, shuffle=True):
    """
    Simple K-fold index generator using NumPy only.
    """
    idx = np.arange(T)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    fold_sizes = np.full(n_splits, T // n_splits, dtype=int)
    fold_sizes[: (T % n_splits)] += 1

    folds = []
    start = 0
    for fs in fold_sizes:
        test = idx[start : start + fs]
        train = np.concatenate([idx[:start], idx[start + fs :]])
        folds.append((train, test))
        start += fs
    return folds


def _standardize_train_apply_test(x_train, x_test):
    """
    Standardize using train mean/std; apply to train and test.
    """
    mu = float(np.mean(x_train))
    sd = float(np.std(x_train))
    if (not np.isfinite(sd)) or sd <= 0:
        sd = 1.0
    return (x_train - mu) / sd, (x_test - mu) / sd


def _unique_transform_train_test(a_tr, p_tr, a_te, p_te):
    """
    Partial out each predictor against the other (with intercept), fit on TRAIN,
    apply the learned linear residualization to TEST.

    a_u = a - [1, p] beta_a   where beta_a fit on TRAIN
    p_u = p - [1, a] beta_p   where beta_p fit on TRAIN
    """
    # a_u
    Xa = np.column_stack([np.ones_like(p_tr), p_tr])
    beta_a = np.linalg.lstsq(Xa, a_tr, rcond=None)[0]
    a_u_tr = a_tr - Xa @ beta_a
    a_u_te = a_te - np.column_stack([np.ones_like(p_te), p_te]) @ beta_a

    # p_u
    Xp = np.column_stack([np.ones_like(a_tr), a_tr])
    beta_p = np.linalg.lstsq(Xp, p_tr, rcond=None)[0]
    p_u_tr = p_tr - Xp @ beta_p
    p_u_te = p_te - np.column_stack([np.ones_like(a_te), a_te]) @ beta_p

    return a_u_tr, p_u_tr, a_u_te, p_u_te


# ------------------------------------------------------------
# CV tuning
# ------------------------------------------------------------
def select_lambda_cv(Y, aff, pow_, lam_grid, n_folds=5, seed=0, shuffle=True, unique=False):
    """
    Choose lambda via K-fold CV minimizing mean squared error (MSE) across voxels.

    Returns:
      best_lam, best_mse, mse_by_lam (dict), n_used_folds
    """
    Y = np.asarray(Y, float)
    aff = np.asarray(aff, float)
    pow_ = np.asarray(pow_, float)

    T = Y.shape[0]
    if T < max(12, n_folds * 2):
        return np.nan, np.nan, {}, 0

    folds = _make_folds(T, n_splits=n_folds, seed=seed, shuffle=shuffle)

    mse_by_lam = {float(lam): [] for lam in lam_grid}
    used = 0

    for tr, te in folds:
        Y_tr = Y[tr]
        Y_te = Y[te]

        # Standardize predictors on TRAIN and apply to TEST
        a_tr, a_te = _standardize_train_apply_test(aff[tr], aff[te])
        p_tr, p_te = _standardize_train_apply_test(pow_[tr], pow_[te])

        if unique:
            a_tr, p_tr, a_te, p_te = _unique_transform_train_test(a_tr, p_tr, a_te, p_te)

        X_tr = np.column_stack([np.ones_like(a_tr), a_tr, p_tr])
        X_te = np.column_stack([np.ones_like(a_te), a_te, p_te])

        # Skip fold if degenerate
        if X_tr.shape[0] < 6 or X_te.shape[0] < 2:
            continue

        used += 1

        for lam in lam_grid:
            B = _ridge_fit_B(X_tr, Y_tr, lam)
            Y_hat = X_te @ B
            mse = float(np.mean((Y_te - Y_hat) ** 2))
            mse_by_lam[float(lam)].append(mse)

    if used == 0:
        return np.nan, np.nan, {}, 0

    # Average across folds
    mean_mse = {lam: float(np.mean(v)) if len(v) > 0 else np.inf for lam, v in mse_by_lam.items()}
    best_lam = min(mean_mse, key=mean_mse.get)
    best_mse = mean_mse[best_lam]

    return float(best_lam), float(best_mse), mean_mse, int(used)


# ------------------------------------------------------------
# Final axis-angle stat using tuned lambda (full data fit)
# ------------------------------------------------------------
def axis_angle_stat(Y, aff, pow_, lam, unique=False):
    """
    Fit on full data (after zscoring predictors over all trials) and compute axis-angle.
    """
    if not np.isfinite(lam):
        return None

    a = zscore(aff, nan_policy="omit")
    p = zscore(pow_, nan_policy="omit")

    # valid rows already filtered upstream; but keep this guard in case
    m = np.isfinite(a) & np.isfinite(p) & np.isfinite(Y).all(axis=1)
    Y = Y[m]
    a = a[m]
    p = p[m]
    if Y.shape[0] < 8:
        return None

    pred_corr = float(np.corrcoef(a, p)[0, 1])

    if unique:
        # unique transform fit on full data (fixed for observed + null)
        Xa = np.column_stack([np.ones_like(p), p])
        beta_a = np.linalg.lstsq(Xa, a, rcond=None)[0]
        a = a - Xa @ beta_a

        Xp = np.column_stack([np.ones_like(a), a])
        beta_p = np.linalg.lstsq(Xp, p, rcond=None)[0]
        p = p - Xp @ beta_p

    X = np.column_stack([np.ones_like(a), a, p])
    B = _ridge_fit_B(X, Y, lam)

    u_aff = B[1]
    u_pow = B[2]
    cosine, angle = _cos_angle(u_aff, u_pow)

    return dict(
        cosine=cosine,
        angle_deg=angle,
        n_trials=int(Y.shape[0]),
        pred_corr=pred_corr,
    )


# ------------------------------------------------------------
# Freedman–Lane circular-shift null (lambda held fixed)
# ------------------------------------------------------------
def freedman_lane_circular_null(Y, aff, pow_, lam, unique=False, n_perm=200, seed=0):
    """
    Reduced model = intercept only (since Y has already been detrended upstream).
    Permute residuals via circular shift along trial axis.
    Fit full model each time and compute cosine between u_aff and u_pow.

    Note: lambda is held fixed (chosen by CV on observed data). This keeps runtime tractable.
    """
    if not np.isfinite(lam):
        return None

    a = zscore(aff, nan_policy="omit")
    p = zscore(pow_, nan_policy="omit")

    m = np.isfinite(a) & np.isfinite(p) & np.isfinite(Y).all(axis=1)
    Y = Y[m]
    a = a[m]
    p = p[m]
    if Y.shape[0] < 12:
        return None

    if unique:
        Xa = np.column_stack([np.ones_like(p), p])
        beta_a = np.linalg.lstsq(Xa, a, rcond=None)[0]
        a = a - Xa @ beta_a

        Xp = np.column_stack([np.ones_like(a), a])
        beta_p = np.linalg.lstsq(Xp, p, rcond=None)[0]
        p = p - Xp @ beta_p

    X = np.column_stack([np.ones_like(a), a, p])

    rng = np.random.default_rng(seed)
    T = Y.shape[0]

    Y_hat = np.mean(Y, axis=0, keepdims=True)  # intercept-only fit
    R = Y - Y_hat

    shifts = rng.integers(1, T, size=int(n_perm))
    null_cos = np.empty(len(shifts), dtype=float)

    for i, s in enumerate(shifts):
        Yp = Y_hat + np.roll(R, int(s), axis=0)
        Bp = _ridge_fit_B(X, Yp, lam)
        c, _ = _cos_angle(Bp[1], Bp[2])
        null_cos[i] = c

    return null_cos


def perm_pvals(obs, null):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(obs):
        return np.nan, np.nan

    p_one = (1.0 + np.sum(null >= obs)) / (null.size + 1.0)  # one-sided (greater)
    mu = float(np.mean(null))
    p_two = (1.0 + np.sum(np.abs(null - mu) >= np.abs(obs - mu))) / (null.size + 1.0)
    return float(p_one), float(p_two)


# ------------------------------------------------------------
# Analysis loop
# ------------------------------------------------------------
cols = [
    "sub_id", "roi",
    "n_trials", "pred_corr",
    "lam_grid", "n_folds", "shuffle_folds", "seed", "n_perm",

    # tuned lambdas + CV performance
    "lam_raw", "cv_mse_raw", "cv_used_folds_raw",
    "lam_uniq", "cv_mse_uniq", "cv_used_folds_uniq",

    # observed
    "raw_cosine", "raw_angle_deg",
    "uniq_cosine", "uniq_angle_deg",

    # null summaries + z + p
    "raw_null_mean", "raw_null_sd", "raw_z", "raw_p_one", "raw_p_two",
    "uniq_null_mean", "uniq_null_sd", "uniq_z", "uniq_p_one", "uniq_p_two",
]

for roi in tqdm(rois, desc="ROIs"):
    for sub_id in tqdm(all_incl_subs, desc=f"{roi}", leave=False):
        if (sub_id, roi) in processed:
            continue

        row = {c: np.nan for c in cols}
        row.update(
            dict(
                sub_id=sub_id,
                roi=roi,
                lam_grid=",".join([str(x) for x in lam_grid]),
                n_folds=n_folds,
                shuffle_folds=bool(shuffle_folds),
                seed=seed,
                n_perm=n_perm,
            )
        )

        try:
            sd = subject_data[sub_id]
            behavior = sd["behavior"]

            if not {"affil_coord", "power_coord", "onset"}.issubset(behavior.columns):
                raise KeyError("Missing required behavior columns.")

            aff = behavior["affil_coord"].to_numpy(float)
            pow_ = behavior["power_coord"].to_numpy(float)
            onsets = behavior["onset"].to_numpy(float)

            betas = sd["roi_betas"][roi]  # (T,V)
            betas_res = regress_out_temporal_trend(betas, onsets, order=2)

            # Filter once (consistent rows for CV, observed, null)
            Yf, afff, powf, T = _filter_rows(betas_res, aff, pow_)
            if T < 16:
                raise ValueError("Not enough valid trials after filtering.")

            # -------------------------
            # CV tuning (raw + unique)
            # -------------------------
            lam_raw, mse_raw, _, used_raw = select_lambda_cv(
                Yf, afff, powf, lam_grid, n_folds=n_folds, seed=seed,
                shuffle=shuffle_folds, unique=False
            )
            lam_uniq, mse_uniq, _, used_uniq = select_lambda_cv(
                Yf, afff, powf, lam_grid, n_folds=n_folds, seed=seed,
                shuffle=shuffle_folds, unique=True
            )

            row["lam_raw"] = lam_raw
            row["cv_mse_raw"] = mse_raw
            row["cv_used_folds_raw"] = used_raw

            row["lam_uniq"] = lam_uniq
            row["cv_mse_uniq"] = mse_uniq
            row["cv_used_folds_uniq"] = used_uniq

            # -------------------------
            # Observed stats with tuned lambda
            # -------------------------
            raw = axis_angle_stat(Yf, afff, powf, lam=lam_raw, unique=False)
            uniq = axis_angle_stat(Yf, afff, powf, lam=lam_uniq, unique=True)

            if raw is None:
                raise ValueError("Axis-angle failed for raw.")

            row["n_trials"] = raw["n_trials"]
            row["pred_corr"] = raw["pred_corr"]
            row["raw_cosine"] = raw["cosine"]
            row["raw_angle_deg"] = raw["angle_deg"]

            if uniq is not None:
                row["uniq_cosine"] = uniq["cosine"]
                row["uniq_angle_deg"] = uniq["angle_deg"]

            # -------------------------
            # Freedman–Lane circular-shift nulls (lambda fixed)
            # -------------------------
            null_raw = freedman_lane_circular_null(
                Yf, afff, powf, lam=lam_raw, unique=False, n_perm=n_perm, seed=seed
            )
            if null_raw is not None:
                mu = float(np.nanmean(null_raw))
                sd0 = float(np.nanstd(null_raw, ddof=1))
                row["raw_null_mean"] = mu
                row["raw_null_sd"] = sd0
                row["raw_z"] = (row["raw_cosine"] - mu) / sd0 if np.isfinite(sd0) and sd0 > 0 else np.nan
                p1, p2 = perm_pvals(row["raw_cosine"], null_raw)
                row["raw_p_one"] = p1
                row["raw_p_two"] = p2

            null_uniq = freedman_lane_circular_null(
                Yf, afff, powf, lam=lam_uniq, unique=True, n_perm=n_perm, seed=seed
            )
            if null_uniq is not None and np.isfinite(row["uniq_cosine"]):
                mu = float(np.nanmean(null_uniq))
                sd0 = float(np.nanstd(null_uniq, ddof=1))
                row["uniq_null_mean"] = mu
                row["uniq_null_sd"] = sd0
                row["uniq_z"] = (row["uniq_cosine"] - mu) / sd0 if np.isfinite(sd0) and sd0 > 0 else np.nan
                p1, p2 = perm_pvals(row["uniq_cosine"], null_uniq)
                row["uniq_p_one"] = p1
                row["uniq_p_two"] = p2

        except Exception:
            # keep NaNs; still write row
            pass

        pd.DataFrame([row], columns=cols).to_csv(
            out_csv,
            mode="a",
            header=not os.path.exists(out_csv),
            index=False,
        )

        processed.add((sub_id, roi))

from utils import *


def _as_2d_array(X):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array (n_trials, n_features). Got {X.shape}")
    return X

def zscore_features_across_trials(X):
    """Z-score each feature across trials (recommended for geometry metrics)."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def effect_vector(X, ypm1):
    """Mean(X | y=+1) - Mean(X | y=-1)."""
    X_pos = X[ypm1 == 1]
    X_neg = X[ypm1 == -1]
    if X_pos.shape[0] < 2 or X_neg.shape[0] < 2:
        raise ValueError("Not enough samples in one class to compute effect vector.")
    return X_pos.mean(axis=0) - X_neg.mean(axis=0)

def cosine_similarity(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def angle_degrees(cos_val):
    cos_val = float(np.clip(cos_val, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))

def axis_separation_affil_power_perm(
    X: np.ndarray,
    dim: np.ndarray,
    onsets: np.ndarray | None = None,
    *,
    zscore: bool = True,
    n_perm: int = 500,
    rng=None,
    min_per_class: int = 10,
    n_split: int = 500,
    pos_label: str = "affil",
    neg_label: str = "power",
    eps: float = 1e-12,
):
    """
    Subject-level affil vs power separation in a single ROI.

    Statistic:
      v = mean(X | affil) - mean(X | power)
      obs = ||v||

    Null:
      permute dim labels within subject; recompute ||v||

    Also computes split-half axis stability (angle between v in two halves).

    Returns dict with:
      obs, z, p_right, null_mean, null_sd,
      split_angle_mean_deg, split_angle_sd_deg,
      n_aff, n_pow
    """
    if rng is None:
        rng = np.random.default_rng()

    X = _as_2d_array(X)
    dim = np.asarray(dim)

    if onsets is not None:
        onsets = np.asarray(onsets)

    # keep only {affil, power} if anything else is present
    keep = (dim == pos_label) | (dim == neg_label)
    if not np.all(keep):
        X = X[keep]
        dim = dim[keep]
        if onsets is not None and onsets.shape[0] == keep.shape[0]:
            onsets = onsets[keep]

    # encode labels
    y = np.where(dim == pos_label, 1, -1)

    n_aff = int(np.sum(y == 1))
    n_pow = int(np.sum(y == -1))
    if min(n_aff, n_pow) < int(min_per_class):
        raise ValueError(f"Too few trials per class: affil={n_aff}, power={n_pow}")

    # z-score features across trials
    Xz = zscore_features_across_trials(X) if zscore else X.astype(float)

    # observed
    v = effect_vector(Xz, y)
    obs = float(np.linalg.norm(v))

    # split-half axis stability (descriptive)
    def _split_half_angle_once():
        idx = rng.permutation(Xz.shape[0])
        a = idx[: len(idx) // 2]
        b = idx[len(idx) // 2 :]

        if (np.sum(y[a] == 1) < 2) or (np.sum(y[a] == -1) < 2) or (np.sum(y[b] == 1) < 2) or (np.sum(y[b] == -1) < 2):
            return np.nan

        v_a = effect_vector(Xz[a], y[a])
        v_b = effect_vector(Xz[b], y[b])
        return angle_degrees(cosine_similarity(v_a, v_b))

    angles = np.array([_split_half_angle_once() for _ in range(int(n_split))], dtype=float)
    angles = angles[np.isfinite(angles)]
    split_angle_mean = float(np.mean(angles)) if angles.size else np.nan
    split_angle_sd   = float(np.std(angles, ddof=1)) if angles.size > 1 else np.nan

    # permutation null on ||v||
    null = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        y_perm = rng.permutation(y)
        v_perm = effect_vector(Xz, y_perm)
        null[i] = np.linalg.norm(v_perm)

    null = null[np.isfinite(null)]
    if null.size < max(50, int(n_perm * 0.2)):
        raise ValueError("Too few finite permutations to estimate null reliably.")

    null_mu = float(null.mean())
    null_sd = float(null.std(ddof=1)) if null.size > 1 else np.nan

    z = (obs - null_mu) / (null_sd + eps) if np.isfinite(null_sd) else np.nan

    # right-tailed p (since obs = ||v|| >= 0)
    p_right = (np.sum(null >= obs) + 1.0) / (null.size + 1.0)

    return dict(
        obs=obs,
        z=z,
        p_perm_right=float(p_right),
        null_mean=null_mu,
        null_sd=float(null_sd) if np.isfinite(null_sd) else np.nan,
        split_angle_mean_deg=split_angle_mean,
        split_angle_sd_deg=split_angle_sd,
        n_aff=n_aff,
        n_pow=n_pow,
        zscore=bool(zscore),
        n_perm=int(n_perm),
        min_per_class=int(min_per_class),
        n_split=int(n_split),
    )


results_dir = "../results/roi_analysis/"
os.makedirs(results_dir, exist_ok=True)
out_csv = os.path.join(results_dir, "axis-separation.csv")

if os.path.exists(out_csv):
    results_existing = pd.read_csv(out_csv)
    if ("sub_id" in results_existing.columns) and ("roi" in results_existing.columns):
        processed = set(zip(results_existing["sub_id"].astype(str), results_existing["roi"].astype(str)))
    else:
        processed = set()
else:
    processed = set()

# same data source as your template
subject_data = load_pickle("../analyses/lsa_decision_spm/subject_data_tavares-striatum.pkl")

# parameters 
n_perm = 50
min_per_class = 10
zscore = True
n_split = 50

all_incl_subs = incl_subs + incl_subs_tavares
rois = list(subject_data[18002]['roi_betas'].keys())

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
            "p_perm_right": np.nan,
            "split_angle_mean_deg": np.nan,
            "split_angle_sd_deg": np.nan,
            "n_aff": np.nan,
            "n_pow": np.nan,
            "zscore": bool(zscore),
            "n_perm": int(n_perm),
            "min_per_class": int(min_per_class),
            "n_split": int(n_split),
        }

        try:
            betas = subject_data[sub_id]["roi_betas"][roi]
            beh   = subject_data[sub_id]["behavior"]

            # trialwise inputs
            X      = np.asarray(betas)
            dim    = np.asarray(beh["dimension"].to_numpy() if hasattr(beh["dimension"], "to_numpy") else beh["dimension"])
            onsets = np.asarray(beh["onset"].to_numpy() if hasattr(beh["onset"], "to_numpy") else beh["onset"], float)

            # drop neutral trials (63 -> 60) for consistency with your other analyses
            X      = drop_neutral_trials(X)
            dim    = drop_neutral_trials(dim)
            onsets = drop_neutral_trials(onsets)

            seed = stable_seed("axis_separation", sub_key, roi_key, n_perm, min_per_class, zscore, n_split)
            rng = np.random.default_rng(seed)

            res = axis_separation_affil_power_perm(
                X,
                dim,
                onsets=onsets,
                zscore=zscore,
                n_perm=n_perm,
                rng=rng,
                min_per_class=min_per_class,
                n_split=n_split,
                pos_label="affil",
                neg_label="power",
            )

            row.update({
                "obs": res.get("obs", np.nan),
                "null_mean": res.get("null_mean", np.nan),
                "null_sd": res.get("null_sd", np.nan),
                "z": res.get("z", np.nan),
                "p_perm_right": res.get("p_perm_right", np.nan),
                "split_angle_mean_deg": res.get("split_angle_mean_deg", np.nan),
                "split_angle_sd_deg": res.get("split_angle_sd_deg", np.nan),
                "n_aff": res.get("n_aff", np.nan),
                "n_pow": res.get("n_pow", np.nan),
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

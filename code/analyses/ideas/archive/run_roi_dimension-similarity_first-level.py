from utils import *

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

results_dir = "../results/roi_analysis/"
os.makedirs(results_dir, exist_ok=True)

out_csv = os.path.join(results_dir, "dimension-similarity.csv")
if os.path.exists(out_csv):
    results_existing = pd.read_csv(out_csv)
    processed = set(zip(results_existing["sub_id"], results_existing["roi"]))
else:
    processed = set()

# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------

subject_data = load_pickle("../analyses/lsa_decision_spm/subject_data_tavares-striatum.pkl")

# Dimension labels (optionally drop any non-affil/power trials, e.g., neutral)
dim_all = decision_trials["dimension"].to_numpy()
keep_mask = np.isin(dim_all, ["affil", "power"])
dim_labels = dim_all[keep_mask]
T = len(dim_labels)

tri_i, tri_j = np.triu_indices(T, k=1)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def within_between_from_corr(corr_mat, labels):
    """
    corr_mat: (T, T) trial-by-trial Pearson correlation matrix
    labels:   (T,) dimension labels per trial (e.g., 'affil'/'power')

    Returns:
      within: mean correlation among same-dimension trial pairs (affil-affil and power-power)
      between: mean correlation among different-dimension trial pairs (affil-power)
      diff: within - between
    """
    r_upper = corr_mat[tri_i, tri_j]
    same_dim = (labels[tri_i] == labels[tri_j])

    within = np.nanmean(r_upper[same_dim])
    between = np.nanmean(r_upper[~same_dim])
    diff = within - between
    return within, between, diff

def circular_shift_null_diff(corr_mat, labels, n_perm=100, seed=0):
    """
    Circular-shift the labels to preserve temporal structure, recompute diff each time.
    Returns:
      null_diffs: (n_perm,) null distribution of within-between differences
    """
    rng = np.random.default_rng(seed)
    Tloc = len(labels)
    shifts = rng.integers(1, Tloc, size=n_perm)

    null_diffs = np.empty(n_perm, dtype=float)
    for k, s in enumerate(shifts):
        perm_labels = np.roll(labels, s)
        _, _, d = within_between_from_corr(corr_mat, perm_labels)
        null_diffs[k] = d
    return null_diffs

# ------------------------------------------------------------
# Analysis loop
# ------------------------------------------------------------

n_perm = 50
seed = 0
all_incl_subs = incl_subs + incl_subs_tavares
rois = list(subject_data[18002]['roi_betas'].keys())

cols = ["sub_id", "roi", "within", "between", "diff", "null_mean", "null_sd", "z"]
for roi in tqdm(rois, desc="ROIs"):
    for sub_id in tqdm(all_incl_subs, desc=f"{roi}", leave=False):
        if (sub_id, roi) in processed:
            continue

        row = {c: np.nan for c in cols}
        row["sub_id"] = sub_id
        row["roi"] = roi

        try:
            betas = subject_data[sub_id]["roi_betas"][roi]  # (n_trials, n_voxels)

            # Align trials to the affil/power subset if needed
            if betas.shape[0] == len(dim_all):
                betas = betas[keep_mask]
            elif betas.shape[0] != T:
                raise ValueError(
                    f"Trial count mismatch: betas has {betas.shape[0]}, "
                    f"expected {T} (or {len(dim_all)} before filtering)"
                )

            # Trial-by-trial multivoxel pattern similarity matrix (Pearson r across voxels)
            corr_mat = np.corrcoef(betas)

            # Observed within/between and difference
            within, between, diff = within_between_from_corr(corr_mat, dim_labels)

            # Permutation null (circular shifts of labels)
            null_diffs = circular_shift_null_diff(
                corr_mat,
                dim_labels,
                n_perm=n_perm,
                seed=seed,
            )

            null_mean = np.nanmean(null_diffs)
            null_sd = np.nanstd(null_diffs, ddof=1)
            z = (diff - null_mean) / null_sd if np.isfinite(null_sd) and null_sd > 0 else np.nan

            row.update({
                "within": within,
                "between": between,
                "diff": diff,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "z": z,
            })

        except Exception:
            pass

        pd.DataFrame([row], columns=cols).to_csv(
            out_csv,
            mode="a",
            header=not os.path.exists(out_csv),
            index=False,
        )

        processed.add((sub_id, roi))

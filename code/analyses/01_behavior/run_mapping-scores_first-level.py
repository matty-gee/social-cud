''' Run mapping scores and save to file '''
from utils import *

# mapping scores
def mapping_score_distance_z(
    beh,
    sub,
    n_perm=100,
    seed=2026,
):
    """
    Mean character-wise Euclidean distance, z-scored against
    a subject-specific permutation null (character shuffling).

    Returns: z-score (higher = better mapping)
    """
    rng = np.random.default_rng(seed)

    # observed error
    obs_err = np.mean(np.linalg.norm(beh - sub, axis=1))

    # permutation null
    null_errs = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(len(sub))
        null_errs[i] = np.mean(np.linalg.norm(beh - sub[perm], axis=1))

    mu = null_errs.mean()
    sd = null_errs.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return (mu - obs_err) / sd

def mapping_score_rsa_tau_z(
    beh,
    sub,
    n_perm=100,
    seed=2026,
):
    """
    Mantel-style RSA:
      - observed = Kendall's tau between pairwise Euclidean distance vectors
      - null     = permute character labels (shuffle rows of `sub`), recompute tau
      - z-score  = (obs_tau - mean(null_tau)) / sd(null_tau)

    Returns: z-score (higher = better mapping)
    """
    rng = np.random.default_rng(seed)

    beh = np.asarray(beh, float)
    sub = np.asarray(sub, float)
    if beh.shape != sub.shape or beh.ndim != 2:
        raise ValueError(f"beh and sub must both be (n_char, n_dim) with same shape; got {beh.shape} vs {sub.shape}")

    n = beh.shape[0]
    if n < 3:
        # Need >=3 items for a non-trivial distance structure
        return np.nan

    # observed tau
    d_beh = pdist(beh, metric="euclidean")
    d_sub = pdist(sub, metric="euclidean")
    obs_tau, _ = kendalltau(d_beh, d_sub)
    if not np.isfinite(obs_tau):
        return np.nan

    # permutation null (Mantel-style: shuffle labels of one set)
    null_taus = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(n)
        d_sub_perm = pdist(sub[perm], metric="euclidean")
        tau_i, _ = kendalltau(d_beh, d_sub_perm)
        null_taus[i] = tau_i

    mu = np.nanmean(null_taus)
    sd = np.nanstd(null_taus, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan

    return (obs_tau - mu) / sd

def compute_mapping_scores(
    df,
    characters,
    method,
    n_perm=1000,
):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        One row per subject.
    characters : list[str]
        Character name suffixes.
    method : {"distance_z", "rsa", "procrustes"}
    sub_id_col : str
        Subject identifier column.
    """

    scores = []

    for _, row in df.iterrows():
        beh = []
        sub = []

        for ch in characters:
            beh.append([
                row[f"affil_mean_{ch}"],
                row[f"power_mean_{ch}"]
            ])
            sub.append([
                row[f"dots_affil_{ch}"],
                row[f"dots_power_{ch}"]
            ])

        beh = np.asarray(beh)
        sub = np.asarray(sub)

        if method == "distance_z":
            score = mapping_score_distance_z(beh, sub, n_perm=n_perm)
        elif method == "rsa":
            score = mapping_score_rsa_tau_z(beh, sub)
        else:
            raise ValueError(f"Unknown method: {method}")

        scores.append({
            'sub_id': row['sub_id'],
            "mapping_score": score
        })

    return pd.DataFrame(scores)



results_dir = '../results/behavior/'
methods     = ["distance_z", "rsa"]
dfs_inlab, dfs_online = [], []
for method in methods:

    # in-lab data
    ms = compute_mapping_scores(
        data,
        characters=CHARACTERS,
        method=method,
        n_perm=100,
    ).copy()
    ms = ms.rename(columns={"mapping_score": f"mapping_score_{method}"})
    dfs_inlab.append(ms)

    # online data
    ms_on = compute_mapping_scores(
        data_online,
        characters=CHARACTERS,
        method=method,
        n_perm=100,
    ).copy()
    ms_on = ms_on.rename(columns={"mapping_score": f"mapping_score_{method}"})
    dfs_online.append(ms_on)

# merge across methods (outer to avoid losing rows if one method is missing some)
mapping_scores = dfs_inlab[0]
for df in dfs_inlab[1:]:
    merge_keys = [c for c in mapping_scores.columns if c in df.columns and not c.startswith("mapping_score_")]
    mapping_scores = mapping_scores.merge(df, on=merge_keys, how="outer")

mapping_scores_online = dfs_online[0]
for df in dfs_online[1:]:
    merge_keys = [c for c in mapping_scores_online.columns if c in df.columns and not c.startswith("mapping_score_")]
    mapping_scores_online = mapping_scores_online.merge(df, on=merge_keys, how="outer")

# save
mapping_scores.to_csv(f"{results_dir}/mapping_scores.csv", index=False)
mapping_scores_online.to_csv(f"{results_dir}/mapping_scores_online.csv", index=False)

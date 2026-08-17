from utils import *  
from scipy.stats import spearmanr, rankdata
from scipy.stats import wilcoxon  #


# ----------------------------
# Confusion matrices
# ----------------------------

def memory_confusions_by_subject(
    data: pd.DataFrame,
    *,
    normalize: bool = True,   # row-normalize counts -> P(response | true)
    symmetrize: bool = True,  # symmetric "mutual confusion" score
):
    """
    Build per-subject confusion matrices from memory_01_* ... memory_30_* columns.

    Base matrix is counts: C[i,j] = # times true=i, response=j.

    If normalize=True:
        P[i,j] = C[i,j] / sum_j C[i,j]   (row-stochastic; P(response | true))

    If symmetrize=True:
        S[i,j] = 0.5 * (P[i,j] + P[j,i])  (symmetric mutual-confusion score)
      - If normalize=False, symmetrizes counts: 0.5*(C + C.T)

    Note: when normalize=True and symmetrize=True, output is a symmetric *score*,
    not a conditional probability matrix.
    """
    characters = list(CHARACTERS)
    char_to_i = {c: i for i, c in enumerate(characters)}
    K = len(characters)

    pat = re.compile(r"^memory_(\d{2})_(.+)$")
    mem = []
    for col in data.columns:
        m = pat.match(col)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= 30:
                mem.append((idx, col, m.group(2)))
    if not mem:
        raise ValueError("No columns matching memory_01_* ... memory_30_* were found in `data`.")

    mem.sort(key=lambda x: x[0])
    memory_cols = [c for _, c, _ in mem]
    true_labels = [t for _, _, t in mem]

    valid = np.array([t in char_to_i for t in true_labels], dtype=bool)
    if not valid.all():
        bad = [(memory_cols[i], true_labels[i]) for i, ok in enumerate(valid) if not ok]
        raise ValueError(f"Some true labels (from col suffix) not in CHARACTERS: {bad[:10]}")

    true_idx = np.array([char_to_i[t] for t in true_labels], dtype=int)

    resp_df = data.loc[:, memory_cols]
    sub_ids = data["sub_id"].to_numpy()

    conf_by_sub = {}
    for row_i, sub_id in enumerate(sub_ids):
        resp_row = resp_df.iloc[row_i].to_numpy()

        resp_idx = np.full(len(memory_cols), -1, dtype=int)
        for j, r in enumerate(resp_row):
            if pd.isna(r):
                continue
            r_str = str(r)
            if r_str in char_to_i:
                resp_idx[j] = char_to_i[r_str]
            # else ignore unknown responses

        ok = resp_idx >= 0

        C = np.zeros((K, K), dtype=float)
        for t_i, r_i in zip(true_idx[ok], resp_idx[ok]):
            C[t_i, r_i] += 1.0

        if normalize:
            row_sums = C.sum(axis=1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                P = np.divide(C, row_sums, out=np.zeros_like(C), where=row_sums > 0)
        else:
            P = C

        if symmetrize:
            P = 0.5 * (P + P.T)

        conf_by_sub[sub_id] = P

    return conf_by_sub

def asymmetry_matrix(C):
    """
    C: (K, K) confusion counts (directed), diagonal can be 0.
    Returns:
      A: (K, K) antisymmetric matrix where A[i,j] = C[i,j] - C[j,i]
    """
    C = np.asarray(C, float)
    return C - C.T

def asymmetry_magnitude(C, *, eps=1e-12):
    """
    Returns a scalar in [0, 1] (roughly) measuring how directional the confusions are.
    0 means perfectly symmetric, larger means more directional.
    Uses L1 norm of antisymmetric part normalized by total off-diagonal mass.
    """
    C = np.asarray(C, float)
    K = C.shape[0]
    off = ~np.eye(K, dtype=bool)

    A = C - C.T
    num = np.sum(np.abs(A)[off]) / 2.0     # each unordered pair counted twice in A
    den = np.sum(C[off]) + eps
    return float(num / den)

# ----------------------------
# RDM utilities
# ----------------------------

def _coords_to_rdm(coords: np.ndarray) -> np.ndarray:
    diffs = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diffs**2, axis=2))

def dots_rdm_by_subject(data: pd.DataFrame):
    def dots_coords_by_subject(data: pd.DataFrame):
        characters = list(CHARACTERS)

        pat = re.compile(r"^dots_(affil|power)_(.+)$")
        affil_col = {}
        "power_coord" = {}

        for col in data.columns:
            m = pat.match(col)
            if not m:
                continue
            kind, char = m.group(1), m.group(2)
            if char in characters:
                if kind == "affil":
                    affil_col[char] = col
                else:
                    "power_coord"[char] = col

        missing = []
        for char in characters:
            if char not in affil_col:
                missing.append(f"dots_affil_{char}")
            if char not in "power_coord":
                missing.append(f"dots_power_{char}")
        if missing:
            raise ValueError(f"Missing dots columns for some CHARACTERS. First few: {missing[:10]}")

        sub_ids = data["sub_id"].to_numpy()
        coords_by_sub = {}

        for i, sub_id in enumerate(sub_ids):
            coords = np.full((len(characters), 2), np.nan, dtype=float)
            for c, char in enumerate(characters):
                coords[c, 0] = float(data.iloc[i][affil_col[char]])
                coords[c, 1] = float(data.iloc[i]["power_coord"[char]])
            coords_by_sub[sub_id] = coords

        return coords_by_sub

    coords_by_sub = dots_coords_by_subject(data)
    rdm_by_sub = {sub_id: _coords_to_rdm(coords) for sub_id, coords in coords_by_sub.items()}
    return rdm_by_sub

def beh_rdm_by_subject(data: pd.DataFrame):
    def beh_coords_by_subject(data: pd.DataFrame):
        characters = list(CHARACTERS)

        pat = re.compile(r"^(affil_mean|power_mean)_(.+)$")
        affil_col = {}
        power_col = {}

        for col in data.columns:
            m = pat.match(col)
            if not m:
                continue
            kind, char = m.group(1), m.group(2)
            if char in characters:
                if kind == "affil_mean":
                    affil_col[char] = col
                else:
                    power_col[char] = col

        missing = []
        for char in characters:
            if char not in affil_col:
                missing.append(f"affil_mean_{char}")
            if char not in power_col:
                missing.append(f"power_mean_{char}")
        if missing:
            raise ValueError(f"Missing behavior columns for some CHARACTERS. First few: {missing[:10]}")

        sub_ids = data["sub_id"].to_numpy()
        coords_by_sub = {}

        for i, sub_id in enumerate(sub_ids):
            coords = np.full((len(characters), 2), np.nan, dtype=float)
            for c, char in enumerate(characters):
                coords[c, 0] = float(data.iloc[i][affil_col[char]])
                coords[c, 1] = float(data.iloc[i][power_col[char]])
            coords_by_sub[sub_id] = coords

        return coords_by_sub

    coords_by_sub = beh_coords_by_subject(data)
    rdm_by_sub = {sub_id: _coords_to_rdm(coords) for sub_id, coords in coords_by_sub.items()}
    return rdm_by_sub

def flatten_offdiag(M: np.ndarray) -> np.ndarray:
    """Vectorize all off-diagonal entries (directed pairs i->j)."""
    M = np.asarray(M)
    K = M.shape[0]
    mask = ~np.eye(K, dtype=bool)
    return M[mask].astype(float)

def _vectorize_rdm(M: np.ndarray, *, vectorize: str) -> np.ndarray:
    if vectorize == "upper":      # symmetric (unique pairs)
        return np.asarray(flatten_upper_tri(M), float)
    if vectorize == "offdiag":    # directed (i->j and j->i both included)
        return flatten_offdiag(M)
    raise ValueError("vectorize must be one of {'upper','offdiag'}")


# ----------------------------
# Partial RSA (controlling for onset RDM)
# ----------------------------

def mean_onset_rdm(decision_trials: pd.DataFrame):
    """
    Character-level control RDM: |mean_onset_i - mean_onset_j| in CHARACTERS order.
    Expects columns: 'onset' and 'character_role_name' (or adjust below).
    """
    mean_onsets = decision_trials.groupby("character_role_name")["onset"].mean().to_dict()
    mean_onsets = {str(k).lower(): v for k, v in mean_onsets.items()}

    onset_vec = np.array([mean_onsets.get(char, np.nan) for char in CHARACTERS], float)
    rdm = np.abs(onset_vec[:, None] - onset_vec[None, :])
    return rdm

def _residualize(y, X):
    """OLS residuals of y on X (with intercept)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return y - Xd @ beta

def subjectwise_partial_rsa_conf_vs_rdm(
    conf_by_sub: dict,
    rdm_by_sub: dict,
    onset_rdm: np.ndarray,
    *,
    vectorize: str = "upper",   # 'upper' for symmetric; 'offdiag' for directed
):
    """
    Rank-based partial RSA controlling for onset_rdm.

    If vectorize='upper':
        uses upper triangle (unique undirected pairs) -- appropriate for symmetric matrices.
    If vectorize='offdiag':
        uses all off-diagonal entries (directed pairs) -- appropriate for non-symmetric confusion matrices.
    """
    z_full = _vectorize_rdm(onset_rdm, vectorize=vectorize)

    sub_ids = sorted(set(conf_by_sub) & set(rdm_by_sub))
    rows = []

    for sub_id in sub_ids:
        y_full = _vectorize_rdm(conf_by_sub[sub_id], vectorize=vectorize)
        x_full = _vectorize_rdm(rdm_by_sub[sub_id],  vectorize=vectorize)

        ok = np.isfinite(y_full) & np.isfinite(x_full) & np.isfinite(z_full)
        y = y_full[ok]
        x = x_full[ok]
        z = z_full[ok]

        n = int(ok.sum())
        if n < 3:
            rows.append({"sub_id": sub_id, "r_partial": np.nan, "n_pairs": n})
            continue

        yr = rankdata(y, method="average")
        xr = rankdata(x, method="average")
        zr = rankdata(z, method="average")

        ry = _residualize(yr, zr[:, None])
        rx = _residualize(xr, zr[:, None])

        r = float(np.corrcoef(ry, rx)[0, 1])
        rows.append({"sub_id": sub_id, "r_partial": r, "n_pairs": n})

    return pd.DataFrame(rows)

def _prep_partial(df: pd.DataFrame, score_name: str, n_name: str) -> pd.DataFrame:
    df = df.copy()
    if not {"sub_id", "r_partial", "n_pairs"}.issubset(df.columns):
        raise ValueError(f"Expected columns ['sub_id','r_partial','n_pairs']; got {list(df.columns)}")
    return df.rename(columns={"r_partial": score_name, "n_pairs": n_name})[["sub_id", score_name, n_name]]


# ----------------------------
# Split-half reliability
# ----------------------------

def get_memory_cols(data: pd.DataFrame) -> list[str]:
    pat = re.compile(r"^memory_(\d{2})_(.+)$")
    mem = []
    for col in data.columns:
        m = pat.match(col)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= 30:
                mem.append((idx, col))
    if not mem:
        raise ValueError("No columns matching memory_01_* ... memory_30_* were found.")
    mem.sort(key=lambda x: x[0])
    return [c for _, c in mem]

def spearman_brown(r: float) -> float:
    """Spearman–Brown prophecy formula for split-half reliability."""
    if not np.isfinite(r) or r <= -1:
        return np.nan
    return float((2 * r) / (1 + r))

def split_half_rsa_reliability(
    data: pd.DataFrame,
    *,
    rdm_by_sub: dict,
    onset_rdm: np.ndarray,
    conf_kwargs: dict,
    vectorize: str = "upper",
    split: str = "odd_even",   # 'odd_even' or 'random'
    n_splits: int = 200,       # used if split='random'
    rng: int = 0,
) -> pd.DataFrame:
    """
    Split-half reliability of subject-level RSA scores.

    Returns a DataFrame with one row per split:
      - r_pearson across subjects (halfA vs halfB)
      - r_spearman across subjects
      - Spearman-Brown corrected versions
    """
    mem_cols = get_memory_cols(data)
    n_trials = len(mem_cols)
    if n_trials < 6:
        raise ValueError(f"Too few memory columns ({n_trials}) for split-half reliability.")

    rng = np.random.default_rng(rng)
    idx = np.arange(n_trials)

    splits = []
    if split == "odd_even":
        a = idx[::2]
        b = idx[1::2]
        splits = [(a, b)]
    elif split == "random":
        half = n_trials // 2
        for _ in range(n_splits):
            perm = rng.permutation(idx)
            a = perm[:half]
            b = perm[half:2*half]  # drop 1 if odd number of trials
            splits.append((a, b))
    else:
        raise ValueError("split must be one of {'odd_even','random'}")

    rows = []
    for s_i, (a_idx, b_idx) in enumerate(splits):
        cols_a = [mem_cols[i] for i in a_idx]
        cols_b = [mem_cols[i] for i in b_idx]

        df_a = data.loc[:, ["sub_id"] + cols_a]
        df_b = data.loc[:, ["sub_id"] + cols_b]

        conf_a = memory_confusions_by_subject(df_a, **conf_kwargs)
        conf_b = memory_confusions_by_subject(df_b, **conf_kwargs)

        rsa_a = subjectwise_partial_rsa_conf_vs_rdm(conf_a, rdm_by_sub, onset_rdm, vectorize=vectorize)
        rsa_b = subjectwise_partial_rsa_conf_vs_rdm(conf_b, rdm_by_sub, onset_rdm, vectorize=vectorize)

        rsa_a = rsa_a.rename(columns={"r_partial": "r_a"})
        rsa_b = rsa_b.rename(columns={"r_partial": "r_b"})

        m = rsa_a.merge(rsa_b, on="sub_id", how="inner")
        x = m["r_a"].to_numpy(float)
        y = m["r_b"].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)

        n = int(ok.sum())
        if n < 3:
            r_p = np.nan
            r_s = np.nan
        else:
            r_p = float(np.corrcoef(x[ok], y[ok])[0, 1])
            r_s = float(spearmanr(x[ok], y[ok]).correlation)

        rows.append({
            "split_i": s_i,
            "n_subjects": n,
            "r_pearson": r_p,
            "r_spearman": r_s,
            "r_pearson_sb": spearman_brown(r_p),
            "r_spearman_sb": spearman_brown(r_s),
        })

    return pd.DataFrame(rows)

# ----------------------------
# Leave-one-pair-out partial RSA (symmetric case)
# ----------------------------

def leave_one_pair_out_partial_rsa_upper(
    conf_by_sub: dict,
    rdm_by_sub: dict,
    onset_rdm: np.ndarray,
):
    """
    Leave-one-(unordered)-pair-out influence test for the *symmetric/upper-triangle* partial RSA.
    Returns one row per omitted character pair with the group median/mean RSA after dropping it.
    """
    K = onset_rdm.shape[0]
    iu = np.triu_indices(K, k=1)  # matches utils.flatten_upper_tri order
    pair_chars = [(CHARACTERS[i], CHARACTERS[j]) for i, j in zip(iu[0], iu[1])]

    z_full = flatten_upper_tri(onset_rdm)
    sub_ids = sorted(set(conf_by_sub) & set(rdm_by_sub))
    n_pairs = len(z_full)

    def _score_for_mask(pair_mask):
        scores = []
        for sub_id in sub_ids:
            y_full = flatten_upper_tri(conf_by_sub[sub_id])
            x_full = flatten_upper_tri(rdm_by_sub[sub_id])

            ok = (
                np.isfinite(y_full)
                & np.isfinite(x_full)
                & np.isfinite(z_full)
                & pair_mask
            )
            if ok.sum() < 3:
                continue

            yr = rankdata(y_full[ok], method="average")
            xr = rankdata(x_full[ok], method="average")
            zr = rankdata(z_full[ok], method="average")

            ry = _residualize(yr, zr[:, None])
            rx = _residualize(xr, zr[:, None])

            # guard against degenerate vectors
            if np.allclose(ry, ry[0]) or np.allclose(rx, rx[0]):
                continue

            scores.append(float(np.corrcoef(ry, rx)[0, 1]))

        scores = np.asarray(scores, float)
        return {
            "n_subjects": int(np.isfinite(scores).sum()),
            "median_r_partial": float(np.nanmedian(scores)),
            "mean_r_partial": float(np.nanmean(scores)),
        }

    # Full model (no drop)
    full = _score_for_mask(np.ones(n_pairs, dtype=bool))

    rows = []
    for k in range(n_pairs):
        mask = np.ones(n_pairs, dtype=bool)
        mask[k] = False
        summ = _score_for_mask(mask)

        c1, c2 = pair_chars[k]
        rows.append({
            "omit_pair_k": k,
            "omit_char_1": c1,
            "omit_char_2": c2,
            "full_median_r_partial": full["median_r_partial"],
            "full_mean_r_partial": full["mean_r_partial"],
            "loo_median_r_partial": summ["median_r_partial"],
            "loo_mean_r_partial": summ["mean_r_partial"],
            "delta_median": summ["median_r_partial"] - full["median_r_partial"],
            "delta_mean": summ["mean_r_partial"] - full["mean_r_partial"],
            "n_subjects": summ["n_subjects"],
        })

    return pd.DataFrame(rows)


# ----------------------------
# Main
# ----------------------------

def main():

    globals_ = globals()
    required = ["data", "data_online", "decision_trials"]
    missing = [k for k in required if k not in globals_]
    if missing:
        raise RuntimeError(
            f"Missing required globals: {missing}. "
            "Define them before running (data, data_online, decision_trials)."
        )

    results_dir = '../results/behavior'
    os.makedirs(results_dir, exist_ok=True)

    onset_rdm = mean_onset_rdm(decision_trials)

    for sample, df in {"online": data_online, "inlab": data}.items():
        
        # RDMs & confusion matrices
        dots_rdm = dots_rdm_by_subject(df)
        beh_rdm  = beh_rdm_by_subject(df)
        conf_sym = memory_confusions_by_subject(df, normalize=True, symmetrize=True)  # mutual confusion
        conf_dir = memory_confusions_by_subject(df, normalize=True, symmetrize=False)  # directed P(resp|true)

        # Overall asymmetry magnitude (directed)
        conf_asym_mag = [asymmetry_magnitude(conf) for _, conf in conf_dir.items()]

        # RSA: symmetrized (upper triangle)
        df_part_dots_sym = subjectwise_partial_rsa_conf_vs_rdm(conf_sym, dots_rdm, onset_rdm, vectorize="upper")
        df_part_beh_sym  = subjectwise_partial_rsa_conf_vs_rdm(conf_sym, beh_rdm,  onset_rdm, vectorize="upper")

        # RSA: directed (off-diagonal)
        df_part_dots_dir = subjectwise_partial_rsa_conf_vs_rdm(conf_dir, dots_rdm, onset_rdm, vectorize="offdiag")
        df_part_beh_dir  = subjectwise_partial_rsa_conf_vs_rdm(conf_dir, beh_rdm,  onset_rdm, vectorize="offdiag")

        out = (
            _prep_partial(df_part_dots_sym, "mem_conf_dots_sym", "n_pairs_dots_sym")
            .merge(_prep_partial(df_part_dots_dir, "mem_conf_dots_dir", "n_pairs_dots_dir"), on="sub_id", how="outer")
            .merge(_prep_partial(df_part_beh_sym,  "mem_conf_beh_sym",  "n_pairs_beh_sym"),  on="sub_id", how="outer")
            .merge(_prep_partial(df_part_beh_dir,  "mem_conf_beh_dir",  "n_pairs_beh_dir"),  on="sub_id", how="outer")
        )
        out = out.assign(mem_conf_asym_mag=conf_asym_mag)
        out_csv = os.path.join(results_dir, f"mem_conf_{sample}_variants.csv")
        out.to_csv(out_csv, index=False)


        # Leave-one-pair-out influence test (symmetric / upper-triangle)
        loo_dots_sym = leave_one_pair_out_partial_rsa_upper(conf_sym, dots_rdm, onset_rdm).assign(
            measure="mem_conf_dots_sym", sample=sample
        )
        loo_beh_sym = leave_one_pair_out_partial_rsa_upper(conf_sym, beh_rdm, onset_rdm).assign(
            measure="mem_conf_beh_sym", sample=sample
        )

        loo = pd.concat([loo_dots_sym, loo_beh_sym], ignore_index=True)
        loo_csv = os.path.join(results_dir, f"mem_conf_{sample}_loo_pair_influence.csv")
        loo.to_csv(loo_csv, index=False)


        # Split-half reliability (odd/even by default)
        rel_dots_sym = split_half_rsa_reliability(
            df,
            rdm_by_sub=dots_rdm,
            onset_rdm=onset_rdm,
            conf_kwargs={"normalize": True, "symmetrize": True},
            vectorize="upper",
            split="odd_even",
        )
        rel_dots_dir = split_half_rsa_reliability(
            df,
            rdm_by_sub=dots_rdm,
            onset_rdm=onset_rdm,
            conf_kwargs={"normalize": True, "symmetrize": False},
            vectorize="offdiag",
            split="odd_even",
        )
        rel_beh_sym = split_half_rsa_reliability(
            df,
            rdm_by_sub=beh_rdm,
            onset_rdm=onset_rdm,
            conf_kwargs={"normalize": True, "symmetrize": True},
            vectorize="upper",
            split="odd_even",
        )
        rel_beh_dir = split_half_rsa_reliability(
            df,
            rdm_by_sub=beh_rdm,
            onset_rdm=onset_rdm,
            conf_kwargs={"normalize": True, "symmetrize": False},
            vectorize="offdiag",
            split="odd_even",
        )

        reliability = pd.concat(
            [
                rel_dots_sym.assign(measure="mem_conf_dots_sym", sample=sample),
                rel_dots_dir.assign(measure="mem_conf_dots_dir", sample=sample),
                rel_beh_sym.assign(measure="mem_conf_beh_sym",   sample=sample),
                rel_beh_dir.assign(measure="mem_conf_beh_dir",   sample=sample),
            ],
            ignore_index=True,
        )

        rel_csv = os.path.join(results_dir, f"mem_conf_{sample}_split_half_reliability.csv")
        reliability.to_csv(rel_csv, index=False)

        print(f"[{sample}] wrote:")
        print(f"  {out_csv}")
        print(f"  {loo_csv}")
        print(f"  {rel_csv}")

if __name__ == "__main__":
    main()

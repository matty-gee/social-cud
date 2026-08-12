from utils import *
import scipy.stats

results_dir = '../results/behavior/'

# control for choice similarities, word counts etc
cos_df   = pd.read_csv('../data/narratives/task-embeddings/decision-options_cosines.csv')
df_trial = (
    cos_df[["decision_num", "cos", "word_count_diff", "word_count_sum"]]
    .merge(
        decision_trials[["decision_num", "onset", "dimension", "character_role_num"]],
        on="decision_num",
        how="inner",
    )
    .sort_values("decision_num")
    .reset_index(drop=True)
)

ROLE_TO_NAME = {
    1: "first",
    2: "second",
    3: "assistant",
    4: "powerful",
    5: "boss",
    6: "neutral",
}

# ---------------------------- model helpers 

def log_model(rt: np.ndarray, X: pd.DataFrame):
    """Assumes multiplicative noise on RT."""
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(np.log(rt), Xc).fit()

def rank_model(rt: np.ndarray, X: pd.DataFrame):
    """Rank-based distribution-free regression (ranks outcome only)."""
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(scipy.stats.rankdata(rt), Xc).fit()

def _zscore_inplace(df: pd.DataFrame, cols, ddof: int = 0):
    """
    Z-score columns in-place and return scaling params.
    If a column has zero variance, it is set to 0.0 and std is recorded as 0.0.
    """
    params = {}
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x, ddof=ddof))
        if not np.isfinite(sd) or sd == 0.0:
            df[c] = 0.0
            params[c] = {"mean": mu, "std": sd}
        else:
            df[c] = (x - mu) / sd
            params[c] = {"mean": mu, "std": sd}
    return params

def make_char_dummies_named(
    df: pd.DataFrame,
    *,
    char_col: str = "character_role_num",
    drop_first: bool = True,
    role_to_name: dict[int, str] = ROLE_TO_NAME,
    prefix: str = "char",
):
    """
    Create role dummies with stable baseline (lowest ordered category)
    but rename columns using role_to_name (e.g., char_2 -> char_second).
    Requires df[char_col] to be categorical with ordered numeric categories.
    """
    d = pd.get_dummies(df[char_col], prefix=prefix, drop_first=drop_first, dtype=float)

    rename = {}
    for col in d.columns:
        # expected: f"{prefix}_{roleint}"
        try:
            role = int(col.split(f"{prefix}_", 1)[1])
        except Exception:
            continue
        nm = role_to_name.get(role, f"role_{role}")
        rename[col] = f"{prefix}_{nm}"

    return d.rename(columns=rename)

def add_onset_x_character(
    df: pd.DataFrame,
    onset_col: str,
    char_dum: pd.DataFrame,
):
    """
    Create onset × dummy interactions for each dummy column already in char_dum.
    """
    onset = pd.to_numeric(df[onset_col], errors="coerce").to_numpy(dtype=float)
    out = {}
    for c in char_dum.columns:
        out[f"{onset_col}_x_{c}"] = onset * char_dum[c].to_numpy(dtype=float)
    return pd.DataFrame(out, index=df.index)

def extract_onset_slopes_by_character_named(
    fit,
    *,
    roles_in_order: list[int],
    onset_col: str,
    role_to_name: dict[int, str] = ROLE_TO_NAME,
    dummy_prefix: str = "char_",
):
    """
    Implied onset slope per character role.
    Baseline = first role in roles_in_order (because drop_first=True with ordered categorical).

      slope(baseline) = beta_onset
      slope(other)    = beta_onset + beta(onset_x_char_<name>)
    """
    base_role = roles_in_order[0]
    base_name = role_to_name.get(base_role, f"role_{base_role}")

    base_beta = float(fit.params.get(onset_col, np.nan))
    rows = []
    for r in roles_in_order:
        nm = role_to_name.get(r, f"role_{r}")
        if r == base_role:
            slope = base_beta
        else:
            dum_name = f"{dummy_prefix}{nm}"         # e.g., "char_boss"
            int_name = f"{onset_col}_x_{dum_name}"   # e.g., "decision_num_x_char_boss"
            slope = base_beta + float(fit.params.get(int_name, 0.0))
        rows.append(
            {
                "role": int(r),
                "character": nm,
                "is_baseline": (r == base_role),
                "onset_slope": float(slope),
            }
        )
    return pd.DataFrame(rows), base_name

# ------------------------- run it across subjects

# main function
def fit_single_subject_rt_and_extract(
    *,
    sub_id,
    behavior: pd.DataFrame,
    df_trial: pd.DataFrame,
    char_col: str = "character_role_num",
    rt_col: str = "reaction_time",
    onset_col: str = "onset",        # keep using onset as the regressor
    dim_col: str = "dimension",
    cos_col: str = "cos",
    wc_sum_col: str = "word_count_sum",
    role_to_name: dict[int, str] = ROLE_TO_NAME,
    zscore_predictors: bool = True,
    return_fits: bool = False,
):
    """
    Uses `onset_col` as the model regressor (default: "onset") but ALWAYS merges
    df_trial onto behavior using "decision_num" (stable integer key).
    """
    EXPECTED_TRIALS_POST_NEUTRAL = 60
    MERGE_KEY = "decision_num"

    # -------------------------- validate merge key --------------------------
    if MERGE_KEY not in behavior.columns:
        raise KeyError(f"behavior is missing required merge key column: '{MERGE_KEY}'")
    if MERGE_KEY not in df_trial.columns:
        raise KeyError(f"df_trial is missing required merge key column: '{MERGE_KEY}'")

    # -------------------------- validate df_trial --------------------------
    needed_trial_cols = [MERGE_KEY, cos_col, wc_sum_col]
    missing = [c for c in needed_trial_cols if c not in df_trial.columns]
    if missing:
        raise KeyError(f"df_trial is missing required columns: {missing}")

    if df_trial[MERGE_KEY].duplicated().any():
        dup = df_trial.loc[df_trial[MERGE_KEY].duplicated(), MERGE_KEY].unique()[:10]
        raise ValueError(
            f"df_trial has duplicate {MERGE_KEY} values (examples: {dup}). "
            "Deduplicate df_trial on decision_num before merging."
        )

    # ---------------------------- merge (ALWAYS on decision_num) ----------------------------
    beh = behavior.copy()

    # drop any stale columns that might already exist in behavior
    for c in [cos_col, wc_sum_col]:
        if c in beh.columns:
            beh = beh.drop(columns=[c])

    # ensure merge key is comparable type
    beh[MERGE_KEY] = pd.to_numeric(beh[MERGE_KEY], errors="coerce")
    dft = df_trial[needed_trial_cols].copy()
    dft[MERGE_KEY] = pd.to_numeric(dft[MERGE_KEY], errors="coerce")

    beh = beh.merge(dft, on=MERGE_KEY, how="inner")

    # ---------------------------- coerce ----------------------------
    beh[rt_col] = pd.to_numeric(beh[rt_col], errors="coerce")

    # onset_col is a regressor; ensure it's numeric and present
    if onset_col not in beh.columns:
        raise KeyError(f"behavior is missing onset_col='{onset_col}' needed for the model.")
    beh[onset_col] = pd.to_numeric(beh[onset_col], errors="coerce")

    beh[cos_col] = pd.to_numeric(beh[cos_col], errors="coerce")
    beh[wc_sum_col] = pd.to_numeric(beh[wc_sum_col], errors="coerce")

    beh[char_col] = pd.to_numeric(beh[char_col], errors="coerce")
    beh = beh[np.isfinite(beh[char_col])].copy()
    beh[char_col] = beh[char_col].astype(int)

    # ---------------------------- drop neutral ----------------------------
    beh = beh[beh[char_col] != 6].copy()

    if len(beh) != EXPECTED_TRIALS_POST_NEUTRAL:
        counts = beh[char_col].value_counts().sort_index().to_dict()
        raise ValueError(
            f"Subject {sub_id}: expected {EXPECTED_TRIALS_POST_NEUTRAL} trials after "
            f"dropping neutral, got {len(beh)}. Counts by role: {counts}"
        )

    # ---------------------------- dimension regressor ----------------------------
    beh["dim_is_power"] = (
        beh[dim_col].astype(str).str.lower() == "power"
    ).astype(float)

    # ---------------------------- RT filtering ----------------------------
    beh = beh[np.isfinite(beh[rt_col]) & (beh[rt_col] > 0)].copy()

    finite_mask = (
        np.isfinite(beh[onset_col]) &
        np.isfinite(beh[cos_col]) &
        np.isfinite(beh[wc_sum_col]) &
        np.isfinite(beh["dim_is_power"])
    )
    beh = beh.loc[finite_mask].copy()

    # ---------------------------- roles ----------------------------
    roles_expected = sorted([r for r in role_to_name if r != 6])
    beh[char_col] = pd.Categorical(
        beh[char_col],
        categories=roles_expected,
        ordered=True,
    )
    roles_in_order = list(beh[char_col].cat.categories)

    if beh[char_col].isna().any():
        raise ValueError(f"Subject {sub_id}: missing expected roles after filtering.")

    # ---------------------------- z-score ----------------------------
    zscore_params = {}
    if zscore_predictors:
        zscore_params = _zscore_inplace(
            beh, [onset_col, cos_col, wc_sum_col]
        )

    rt = beh[rt_col].to_numpy(float)

    # ---------------------------- design matrices ----------------------------
    base_cols = [onset_col, "dim_is_power", cos_col, wc_sum_col]
    X_base = beh[base_cols]

    char_dum = make_char_dummies_named(
        beh, char_col=char_col, drop_first=True
    )

    X1 = pd.concat([X_base, char_dum], axis=1)
    fit1 = log_model(rt, X1)

    X2 = pd.concat(
        [X1, add_onset_x_character(beh, onset_col, char_dum)],
        axis=1,
    )
    fit2 = log_model(rt, X2)

    # ---------------------------- output ----------------------------
    row = {
        "sub_id": sub_id,
        "n_trials": len(beh),
        "r2": float(fit1.rsquared),
        "adj_r2": float(fit1.rsquared_adj),
        "bic": float(fit1.bic),
    }

    for k, v in fit1.params.items():
        row[f"beta_{k}"] = float(v)

    row["intrxn_r2"] = float(fit2.rsquared)
    row["intrxn_adj_r2"] = float(fit2.rsquared_adj)
    row["intrxn_bic"] = float(fit2.bic)

    slopes_df, _ = extract_onset_slopes_by_character_named(
        fit2,
        roles_in_order=roles_in_order,
        onset_col=onset_col,
    )

    for _, r in slopes_df.iterrows():
        row[f"intrxn_beta_onset_{r['character']}"] = float(r["onset_slope"])

    if return_fits:
        return row, {
            "fit_main": fit1,
            "fit_interaction": fit2,
            "behavior": beh,
            "zscore_params": zscore_params,
        }

    return row

# wrappers
def fit_rt_models_from_single_subject(
    *,
    df_trial: pd.DataFrame,
    incl_subs,
    merge_df: pd.DataFrame | None = None,
    on_missing_sub: str = "skip",   # {"skip","raise"}
    verbose: bool = True,
    **fit_kws,                      # forwarded to fit_single_subject_rt_and_extract
) -> pd.DataFrame:
    """
    Minimal multi-subject wrapper around fit_single_subject_rt_and_extract().
    Returns one row per subject (plus optional merge_df columns).

    Notes
    -----
    - If a subject fails for any reason and on_missing_sub="skip", the subject is skipped.
    - If verbose=True, prints the reason each subject is skipped (critical for debugging).
    """
    if on_missing_sub not in {"skip", "raise"}:
        raise ValueError("on_missing_sub must be 'skip' or 'raise'")

    rows = []
    for sub_id in incl_subs:
        # --- load behavior robustly ---
        try:
            beh = load_behavior(sub_id, on_missing="none")
        except Exception as e:
            if on_missing_sub == "raise":
                raise
            if verbose:
                print(f"[RT MODEL] load_behavior failed for {sub_id}: {type(e).__name__}: {e}")
            continue

        if beh is None or len(beh) == 0:
            if on_missing_sub == "raise":
                raise FileNotFoundError(f"No behavior found for sub_id={sub_id}")
            if verbose:
                print(f"[RT MODEL] skipping {sub_id}: behavior missing/empty")
            continue

        # --- fit single-subject model ---
        try:
            row = fit_single_subject_rt_and_extract(
                sub_id=sub_id,
                behavior=beh,
                df_trial=df_trial,
                **fit_kws,
            )
        except Exception as e:
            if on_missing_sub == "raise":
                raise
            if verbose:
                print(f"[RT MODEL] skipping {sub_id}: {type(e).__name__}: {e}")
            continue

        rows.append(row)

    out = pd.DataFrame(rows)

    if merge_df is not None and "sub_id" in merge_df.columns and len(out) > 0:
        out = out.merge(merge_df, on="sub_id", how="left")

    return out

def get_or_fit_rt_betas_single_subject(
    *,
    out_fname: str,
    df_trial: pd.DataFrame,
    incl_subs,
    merge_df: pd.DataFrame | None = None,
    overwrite: bool = False,
    on_missing_sub: str = "skip",
    **fit_kws,
) -> pd.DataFrame:
    """
    Cache-aware wrapper for the new single-subject RT pipeline.
    """
    if os.path.isfile(out_fname) and not overwrite:
        print(f"Loading cached RT betas: {out_fname}")
        return pd.read_csv(out_fname)

    print(f"Fitting RT betas (writing to): {out_fname}")
    df_betas = fit_rt_models_from_single_subject(
        df_trial=df_trial,
        incl_subs=incl_subs,
        merge_df=merge_df,
        on_missing_sub=on_missing_sub,
        **fit_kws,
    )
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    df_betas.to_csv(out_fname, index=False)
    return df_betas

OVERWRITE = True

rt_betas_tavares = get_or_fit_rt_betas_single_subject(
    out_fname=f"{results_dir}/rt-model_betas_tavares.csv",
    df_trial=df_trial,
    incl_subs=incl_subs_tavares,
    merge_df=data_tavares,
    overwrite=OVERWRITE,
)

rt_betas = get_or_fit_rt_betas_single_subject(
    out_fname=f"{results_dir}/rt-model_betas.csv",
    df_trial=df_trial,
    incl_subs=incl_subs,
    merge_df=data,
    overwrite=OVERWRITE,
)

rt_betas_online = get_or_fit_rt_betas_single_subject(
    out_fname=f"{results_dir}/rt-model_betas_online.csv",
    df_trial=df_trial,
    incl_subs=incl_subs_online,
    merge_df=data_online,
    overwrite=OVERWRITE,
)
def clean_roi_betas(X, min_voxels=5):
    """
    Clean one ROI beta matrix.

    X shape: trials × voxels.
    """
    X = np.asarray(X, float)

    if X.ndim != 2:
        return None

    # drop bad voxels
    good_vox = np.isfinite(X).all(axis=0)
    X = X[:, good_vox]

    # drop zero-variance voxels
    if X.shape[1] > 0:
        good_var = np.nanstd(X, axis=0) > 0
        X = X[:, good_var]

    if X.shape[1] < min_voxels:
        return None

    return X

def zscore_safe(x, eps=1e-12):
    x = np.asarray(x, float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / (sd + eps)

def neural_rdv_from_betas(roi_betas, metric="correlation"):
    """
    Neural RDV: pairwise correlation distance between trialwise beta patterns.

    Positive values = more dissimilar neural patterns.
    """
    X = clean_roi_betas(roi_betas)
    if X is None:
        return None
    return pdist(X, metric=metric)

def get_behavior_vector(behavior, col=None, candidates=None, default=None):
    """
    Get a numeric or categorical vector from behavior.
    """
    if col is not None:
        if col not in behavior.columns:
            raise KeyError(f"Column not found in behavior: {col}")
        return behavior[col].to_numpy()

    candidates = candidates or []
    for c in candidates:
        if c in behavior.columns:
            return behavior[c].to_numpy()

    if default is not None:
        return np.asarray(default)

    raise KeyError(f"None of these columns found in behavior: {candidates}")

# ------------------------------------------------------------
# RSA regression fitting
# ------------------------------------------------------------

def make_rsa_rdvs(
    behavior,
    embeddings=None,
    *,
    rsa_covariates=(
        "dimension",
        "scene_num",
        "char_1", "char_2", "char_3", "char_4", "char_5",
        "character_decision_num",
        "choice",
        "semantic_choice",
        "location",
        "location_runmean",
        "location_dispersion",
        "semantic_runmean",
        "semantic_dispersion",
        "time",
        "time_sq",
        "reaction_time",
    ),
    role_values=(1, 2, 3, 4, 5),
    return_trial_features=False,
    eps=1e-12,
):
    """
    Build RSA RDVs from behavior + optional trialwise embeddings.

    Output:
        rdvs : dict[str, np.ndarray]
            Each value is a vectorized upper-triangle RDV with length T*(T-1)/2.

    Optional:
        if return_trial_features=True:
            returns (rdvs, trial_features)

    Main RDVs
    ---------
    Task structure:
      dimension:
          Binary distance between decision dimensions.
          0 = same dimension, 1 = different dimension.

      scene_num:
          Absolute difference in scene number.

      char_1 ... char_5:
          Character-specific same-character distance.
          For char_k: 0 = both trials are character k, 1 = otherwise.
          This matches the user's distance-coded wording.

      samechar_1 ... samechar_5:
          Compatibility alternative.
          For samechar_k: 1 = both trials are character k, 0 = otherwise.

      character:
          Generic character distance.
          0 = same character, 1 = different character.

      same_character:
          Generic same-character similarity.
          1 = same character, 0 = different character.

      character_decision_num:
          Absolute difference in within-character decision number.

    Choice-related:
      choice:
          Cosine distance between signed choice vectors:
          [affil_decision, power_decision].

      semantic_choice:
          Cosine distance between choice embeddings.

      choice_dispersion:
          Absolute difference in character-specific running choice-vector dispersion,
          computed after the current choice as mean cosine distance of prior/current
          choice vectors from the current running mean choice-vector direction.

    Social and semantic space:
      location:
          Euclidean distance between cumulative character-specific affiliation/power
          locations after the current choice.

      location_runmean:
          Euclidean distance between character-specific running mean affiliation/power
          states after the current choice.

      location_dispersion:
          Absolute difference in character-specific running spatial dispersion after
          the current choice, computed as RMS Euclidean distance of prior/current
          affiliation-power states from the current running mean state.

      semantic_runmean:
          Cosine distance between character-specific running mean choice embeddings
          after the current choice.

      semantic_dispersion:
          Absolute difference in character-specific running semantic dispersion after
          the current choice, computed as mean cosine distance of prior/current choice
          embeddings from the current running mean embedding.

    Nuisance:
      time:
          Absolute temporal distance between onsets.

      time_sq:
          Squared temporal distance between onsets: |onset_i - onset_j|^2.

      reaction_time:
          Absolute difference between reaction times.

    Compatibility aliases
    ---------------------
      dimension_diff             -> dimension
      semantic                   -> semantic_choice
      choice_vector              -> choice
      location_euclidean         -> location
      location_runmean_euclidean -> location_runmean
      semantic_content           -> semantic_runmean
      semantic_distance          -> semantic_dispersion
      time_linear / onset        -> time
      time_quadratic / onset_sq  -> time_sq
      different_character        -> character
      character_binary           -> character
      location_variability       -> choice_dispersion
      location_distance          -> choice_dispersion
    """

    # ---------------------------------------------------------------------
    # Basic setup
    # ---------------------------------------------------------------------

    b = behavior.copy().reset_index(drop=True)
    T = len(b)
    if T < 2:
        raise ValueError(f"Need at least 2 trials to make RDVs; got T={T}")

    ii, jj = np.triu_indices(T, k=1)
    n_pairs = len(ii)

    def _require_cols(cols, name):
        missing = [c for c in cols if c not in b.columns]
        if missing:
            raise KeyError(f"{name} requires missing behavior columns: {missing}")

    def _first_existing(candidates, name, required=True):
        for c in candidates:
            if c in b.columns:
                return c
        if required:
            raise KeyError(f"{name} requires one of these columns: {candidates}")
        return None

    def _num_col(candidates, name, required=True, default=None):
        c = _first_existing(candidates, name=name, required=required)
        if c is None:
            if default is None:
                return np.full(T, np.nan, float)
            return np.asarray(default, float)
        return pd.to_numeric(b[c], errors="coerce").to_numpy(float)

    def _pair_abs(x):
        x = np.asarray(x, float)
        out = np.abs(x[ii] - x[jj])
        out[~np.isfinite(x[ii]) | ~np.isfinite(x[jj])] = np.nan
        return out

    def _pair_sq_abs(x):
        d = _pair_abs(x)
        return d ** 2

    def _pair_cat_diff(x):
        s = pd.Series(x).reset_index(drop=True)
        codes, _ = pd.factorize(s, sort=True)
        codes = codes.astype(float)
        codes[codes < 0] = np.nan

        out = (codes[ii] != codes[jj]).astype(float)
        out[~np.isfinite(codes[ii]) | ~np.isfinite(codes[jj])] = np.nan
        return out

    def _pair_euclidean(X):
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[0] != T:
            raise ValueError(f"Expected X to be T × D with T={T}; got {X.shape}")

        diff = X[ii] - X[jj]
        out = np.sqrt(np.nansum(diff ** 2, axis=1))
        ok = np.isfinite(X[ii]).all(axis=1) & np.isfinite(X[jj]).all(axis=1)
        out[~ok] = np.nan
        return out

    def _normalize_rows(X):
        X = np.asarray(X, float)
        U = np.full_like(X, np.nan, dtype=float)
        ok = np.isfinite(X).all(axis=1)
        norms = np.linalg.norm(X, axis=1)
        ok = ok & np.isfinite(norms) & (norms > eps)
        U[ok] = X[ok] / norms[ok, None]
        return U, ok

    def _pair_cosine(X):
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[0] != T:
            raise ValueError(f"Expected X to be T × D with T={T}; got {X.shape}")

        U, ok = _normalize_rows(X)
        out = np.full(n_pairs, np.nan, dtype=float)

        ok_pair = ok[ii] & ok[jj]
        if ok_pair.any():
            sim = np.sum(U[ii[ok_pair]] * U[jj[ok_pair]], axis=1)
            out[ok_pair] = 1.0 - np.clip(sim, -1.0, 1.0)

        return out

    # ---------------------------------------------------------------------
    # Trialwise core variables
    # ---------------------------------------------------------------------

    role_col = _first_existing(
        ["character_role_num", "char_role_num"],
        name="character identity",
        required=True,
    )
    roles = pd.to_numeric(b[role_col], errors="coerce").to_numpy(float)

    # Choice vectors: signed 2D increments.
    _require_cols(["affil_decision", "power_decision"], "choice vectors")
    choice_vec = (
        b[["affil_decision", "power_decision"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )

    # Dimension labels. Prefer explicit labels, otherwise infer from signed choice vector.
    dim_col = _first_existing(
        [
            "dimension",
            "decision_dimension",
            "trial_dimension",
            "choice_dimension",
            "decision_type",
            "trial_type",
        ],
        name="decision dimension",
        required=False,
    )
    if dim_col is not None:
        dimension = b[dim_col].astype(object).to_numpy()
    else:
        dimension = np.full(T, np.nan, dtype=object)
        a = choice_vec[:, 0]
        p = choice_vec[:, 1]
        dimension[np.isfinite(a) & (np.abs(a) > eps)] = "affiliation"
        dimension[np.isfinite(p) & (np.abs(p) > eps)] = "power"

    # Cumulative character-specific social-space locations after current choice.
    # If affil_coord/power_coord already exist, use them. Otherwise compute from decisions.
    if {"affil_coord", "power_coord"}.issubset(b.columns):
        location_state = (
            b[["affil_coord", "power_coord"]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )
    else:
        location_state = np.full((T, 2), np.nan, dtype=float)
        for r in pd.unique(pd.Series(roles).dropna()):
            idx = np.flatnonzero(roles == r)
            Xc = choice_vec[idx]
            location_state[idx] = np.cumsum(Xc, axis=0)

    # Within-character decision number. Prefer explicit, otherwise compute 1..n per character.
    if "character_decision_num" in b.columns:
        character_decision_num = pd.to_numeric(
            b["character_decision_num"], errors="coerce"
        ).to_numpy(float)
    else:
        character_decision_num = np.full(T, np.nan, dtype=float)
        for r in pd.unique(pd.Series(roles).dropna()):
            idx = np.flatnonzero(roles == r)
            character_decision_num[idx] = np.arange(1, len(idx) + 1, dtype=float)

    # Time. Prefer onset; otherwise use decision_num/trial_num/row number.
    time_col = _first_existing(
        ["onset", "decision_num", "trial_num"],
        name="time",
        required=False,
    )
    if time_col is not None:
        time = pd.to_numeric(b[time_col], errors="coerce").to_numpy(float)
    else:
        time = np.arange(T, dtype=float)

    # ---------------------------------------------------------------------
    # Running location statistics
    # ---------------------------------------------------------------------

    location_runmean_state = np.full((T, 2), np.nan, dtype=float)
    location_dispersion_value = np.full(T, np.nan, dtype=float)

    for r in pd.unique(pd.Series(roles).dropna()):
        idx = np.flatnonzero(roles == r)
        Xc = location_state[idx]

        for k, t_idx in enumerate(idx):
            Xk = Xc[: k + 1]
            ok = np.isfinite(Xk).all(axis=1)
            Xk = Xk[ok]

            if Xk.shape[0] == 0:
                continue

            mu = Xk.mean(axis=0)
            location_runmean_state[t_idx] = mu

            # RMS Euclidean spread around current running mean.
            # This is the spatial analogue of a running SD/radius.
            d = np.linalg.norm(Xk - mu, axis=1)
            location_dispersion_value[t_idx] = float(np.sqrt(np.mean(d ** 2)))

    # ---------------------------------------------------------------------
    # Running choice-vector dispersion
    # ---------------------------------------------------------------------
    # This is not the same as location_dispersion.
    # It measures directional variability of choice increments, not spread
    # of accumulated social-space states.

    choice_dispersion_value = np.full(T, np.nan, dtype=float)

    for r in pd.unique(pd.Series(roles).dropna()):
        idx = np.flatnonzero(roles == r)
        Xc = choice_vec[idx]

        for k, t_idx in enumerate(idx):
            Xk = Xc[: k + 1]
            U, ok = _normalize_rows(Xk)
            U = U[ok]

            if U.shape[0] == 0:
                continue
            if U.shape[0] == 1:
                choice_dispersion_value[t_idx] = 0.0
                continue

            mu = U.mean(axis=0)
            nmu = np.linalg.norm(mu)
            if not np.isfinite(nmu) or nmu <= eps:
                # If mean direction cancels perfectly, fallback to mean pairwise cosine distance.
                sims = U @ U.T
                tri = np.triu_indices(U.shape[0], k=1)
                choice_dispersion_value[t_idx] = float(
                    np.mean(1.0 - np.clip(sims[tri], -1.0, 1.0))
                )
            else:
                mu = mu / nmu
                sims = U @ mu
                choice_dispersion_value[t_idx] = float(
                    np.mean(1.0 - np.clip(sims, -1.0, 1.0))
                )

    # ---------------------------------------------------------------------
    # Running semantic statistics
    # ---------------------------------------------------------------------

    semantic_runmean_state = None
    semantic_dispersion_value = None

    if embeddings is not None:
        E = np.asarray(embeddings, float)
        if E.ndim != 2 or E.shape[0] != T:
            raise ValueError(
                f"embeddings must be trials × features with T={T}; got {E.shape}"
            )

        D = E.shape[1]
        semantic_runmean_state = np.full((T, D), np.nan, dtype=float)
        semantic_dispersion_value = np.full(T, np.nan, dtype=float)

        # Normalize trial embeddings first, then compute running semantic centroids.
        # This makes cosine dispersion interpretable.
        U_all, ok_all = _normalize_rows(E)

        for r in pd.unique(pd.Series(roles).dropna()):
            idx = np.flatnonzero(roles == r)

            for k, t_idx in enumerate(idx):
                idx_k = idx[: k + 1]
                Uk = U_all[idx_k]
                ok = ok_all[idx_k]
                Uk = Uk[ok]

                if Uk.shape[0] == 0:
                    continue

                mu = Uk.mean(axis=0)
                nmu = np.linalg.norm(mu)

                if not np.isfinite(nmu) or nmu <= eps:
                    continue

                mu = mu / nmu
                semantic_runmean_state[t_idx] = mu

                sims = Uk @ mu
                semantic_dispersion_value[t_idx] = float(
                    np.mean(1.0 - np.clip(sims, -1.0, 1.0))
                )

    # ---------------------------------------------------------------------
    # RDV builder
    # ---------------------------------------------------------------------

    aliases = {
        # old or alternate names -> new conceptual names
        "dimension_diff": "dimension",
        "semantic": "semantic_choice",
        "choice_vector": "choice",

        "location_euclidean": "location",
        "location_runmean_euclidean": "location_runmean",

        "semantic_content": "semantic_runmean",
        "semantic_distance": "semantic_dispersion",

        "time_linear": "time",
        "onset": "time",
        "time_quadratic": "time_sq",
        "onset_sq": "time_sq",

        "different_character": "character",
        "character_binary": "character",

        # old names that were really choice-vector variability
        "location_variability": "choice_dispersion",
        "location_distance": "choice_dispersion",
    }

    def _both_role_value(r):
        out = ((roles[ii] == r) & (roles[jj] == r)).astype(float)
        out[~np.isfinite(roles[ii]) | ~np.isfinite(roles[jj])] = np.nan
        return out

    def _make_one(cov):
        canonical = aliases.get(cov, cov)

        # -------------------------
        # Task structure
        # -------------------------

        if canonical == "dimension":
            return _pair_cat_diff(dimension)

        if canonical == "scene_num":
            x = _num_col(["scene_num"], name="scene_num", required=True)
            return _pair_abs(x)

        if canonical == "character_decision_num":
            return _pair_abs(character_decision_num)

        if canonical == "character":
            return _pair_cat_diff(roles)

        if canonical == "same_character":
            return 1.0 - _pair_cat_diff(roles)

        if canonical.startswith("char_"):
            # Distance-coded character-specific variable:
            # 0 = both trials are this character, 1 = otherwise.
            r = int(canonical.replace("char_", ""))
            return 1.0 - _both_role_value(r)

        if canonical.startswith("samechar_"):
            # Similarity/dummy-coded compatibility variable:
            # 1 = both trials are this character, 0 = otherwise.
            r = int(canonical.replace("samechar_", ""))
            return _both_role_value(r)

        # -------------------------
        # Choice-related
        # -------------------------

        if canonical == "choice":
            return _pair_cosine(choice_vec)

        if canonical == "semantic_choice":
            if embeddings is None:
                raise ValueError("embeddings is required for semantic_choice")
            return _pair_cosine(np.asarray(embeddings, float))

        if canonical == "choice_dispersion":
            return _pair_abs(choice_dispersion_value)

        # -------------------------
        # Social and semantic space
        # -------------------------

        if canonical == "location":
            return _pair_euclidean(location_state)

        if canonical == "location_runmean":
            return _pair_euclidean(location_runmean_state)

        if canonical == "location_dispersion":
            return _pair_abs(location_dispersion_value)

        if canonical == "semantic_runmean":
            if embeddings is None:
                raise ValueError("embeddings is required for semantic_runmean")
            return _pair_cosine(semantic_runmean_state)

        if canonical == "semantic_dispersion":
            if embeddings is None:
                raise ValueError("embeddings is required for semantic_dispersion")
            return _pair_abs(semantic_dispersion_value)

        # -------------------------
        # Nuisance covariates
        # -------------------------

        if canonical == "time":
            return _pair_abs(time)

        if canonical == "time_sq":
            return _pair_sq_abs(time)

        if canonical == "reaction_time":
            x = _num_col(
                ["reaction_time", "rt", "response_time"],
                name="reaction_time",
                required=True,
            )
            return _pair_abs(x)

        # -------------------------
        # Fallback: numeric behavior column
        # -------------------------

        if canonical in b.columns:
            x = pd.to_numeric(b[canonical], errors="coerce").to_numpy(float)
            return _pair_abs(x)

        raise ValueError(
            f"Unknown RSA covariate: {cov!r} "
            f"(canonicalized to {canonical!r})."
        )

    rdvs = {}

    for cov in rsa_covariates:
        if cov in {"char_dummies", "character_dummies"}:
            for r in role_values:
                rdvs[f"char_{int(r)}"] = _make_one(f"char_{int(r)}")
            continue

        if cov in {"samechar_dummies", "same_character_dummies"}:
            for r in role_values:
                rdvs[f"samechar_{int(r)}"] = _make_one(f"samechar_{int(r)}")
            continue

        rdvs[cov] = _make_one(cov)

    # ---------------------------------------------------------------------
    # Optional trialwise feature output for debugging
    # ---------------------------------------------------------------------

    if return_trial_features:
        trial_features = pd.DataFrame(
            {
                "character_role_num": roles,
                "dimension": dimension,
                "time": time,
                "character_decision_num": character_decision_num,
                "affil_decision": choice_vec[:, 0],
                "power_decision": choice_vec[:, 1],
                "affil_coord": location_state[:, 0],
                "power_coord": location_state[:, 1],
                "affil_runmean": location_runmean_state[:, 0],
                "power_runmean": location_runmean_state[:, 1],
                "location_dispersion": location_dispersion_value,
                "choice_dispersion": choice_dispersion_value,
            }
        )

        if semantic_dispersion_value is not None:
            trial_features["semantic_dispersion"] = semantic_dispersion_value

        return rdvs, trial_features

    return rdvs
    
def fit_rsa_regression(
    neural_rdv,
    design_rdvs,
    *,
    nuisance_rdvs=None,
    pair_mask=None,
    zscore_y=True,
    zscore_x=True,
    zscore_binary=False,
    min_pairs=100,
):
    """
    Multiple-regression RSA:

        neural distance ~ nuisance RDVs + focal RSA RDVs

    Returns beta_<effect> for focal RSA predictors only.

    pair_mask selects which trial pairs are included.
    Example:
        pair_mask = trial_lag >= 4
    """
    nuisance_rdvs = nuisance_rdvs or {}

    y = np.asarray(neural_rdv, float)

    all_names = list(nuisance_rdvs.keys()) + list(design_rdvs.keys())
    all_vecs = [np.asarray(nuisance_rdvs[k], float) for k in nuisance_rdvs]
    all_vecs += [np.asarray(design_rdvs[k], float) for k in design_rdvs]

    valid = np.isfinite(y)
    for v in all_vecs:
        valid &= np.isfinite(v)

    if pair_mask is not None:
        pair_mask = np.asarray(pair_mask, bool)
        if pair_mask.shape[0] != y.shape[0]:
            raise ValueError(
                f"pair_mask length={pair_mask.shape[0]}, "
                f"but neural_rdv length={y.shape[0]}"
            )
        valid &= pair_mask

    n = int(valid.sum())

    out = {"n_pairs": n}
    for name in design_rdvs:
        out[f"beta_{name}"] = np.nan

    if n < min_pairs:
        return out

    yv = y[valid]
    if zscore_y:
        yv = zscore_safe(yv)

    X_cols = []
    for v in all_vecs:
        vv = v[valid].astype(float)

        if np.nanstd(vv, ddof=1) < 1e-12:
            vv = np.zeros_like(vv, dtype=float)
        else:
            unique_vals = np.unique(vv[np.isfinite(vv)])
            is_binary = len(unique_vals) <= 2

            if zscore_x and ((not is_binary) or zscore_binary):
                vv = zscore_safe(vv)

        X_cols.append(vv)

    X = np.column_stack(X_cols)

    model = LinearRegression(fit_intercept=True)
    model.fit(X, yv)

    coefs = dict(zip(all_names, model.coef_))

    for name in design_rdvs:
        out[f"beta_{name}"] = float(coefs[name])

    return out

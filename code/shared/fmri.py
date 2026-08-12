import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn.image as nimg
from sklearn.feature_selection import VarianceThreshold
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiSpheresMasker, NiftiMapsMasker
from nilearn.image import load_img, get_data, new_img_like, math_img, binarize_img
from nilearn.masking import compute_brain_mask, compute_multi_brain_mask
from nilearn.plotting import plot_design_matrix
from nilearn.image import resample_to_img, get_data, math_img, threshold_img, binarize_img, new_img_like, smooth_img
from nilearn.input_data import NiftiMasker
from nilearn.masking import intersect_masks, compute_multi_brain_mask, compute_brain_mask, new_img_like
from nilearn.glm import cluster_level_inference, threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel,  make_second_level_design_matrix, non_parametric_inference
from nilearn.reporting import get_clusters_table
from nilearn import image, plotting, masking, datasets
from nilearn.glm.thresholding import threshold_stats_img


from scipy.spatial.distance import pdist
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nilearn import image as nimg
from nilearn import plotting

#-------------------------------- # fmri info 

# atlas for ROIs
base_dir  = '..'
# atlas_pkl = pd.read_pickle(f'{base_dir}/masks/atlases/Schaefer300_HO-subcort25_1mm.pkl')
affine = np.array([[  -2.0999999,    0.       ,    0.       ,   78.       ],
                    [   0.       ,    2.0999999,    0.       , -112.       ],
                    [   0.       ,    0.       ,    2.0999999,  -70.       ],
                    [   0.       ,    0.       ,    0.       ,    1.       ]])
vox_size = (2.1, 2.1, 2.1)
dims = (75, 90, 74) # shape: (75, 90, 74, 1570)


#-------------------------- helper functions

def load_nifti(nifti_fname):  
    return nib.load(nifti_fname)

def get_nifti_info(nifti):
    ''' return dimensions, voxel size and affine matrix of a nifti '''
    if isinstance(nifti, str): nifti = nib.load(nifti)
    dims = nifti.get_fdata().shape
    vox_size = nifti.header.get_zooms()[:3] # just get xyz
    affine_matrix = nifti.affine
    return dims, vox_size, affine_matrix 

def get_voxels_from_mask(func_img, mask_img, resample_to_func=False, standardize=False):
    '''
        mask_img: 3d nii (ideally already resampled to correct dims)
        sub_img: 4d nii
        returns: array of shape (time_points, voxels)
    '''
    if resample_to_func:
        sub_dims, _, sub_affine = get_nifti_info(func_img)
        masker = NiftiMasker(mask_img=mask_img,
                             target_affine=sub_affine, target_shape=sub_dims[:3],
                             standardize=standardize)
    else:
        masker = NiftiMasker(mask_img=mask_img, standardize=standardize)
    return masker.fit_transform(func_img)


#-------------------------- preprocessing 

def regress_out_temporal_trend(
    X: np.ndarray,
    onsets: np.ndarray,
    *,
    order: int = 1,
    center_onsets: bool = True,
    scale_onsets: bool = True,
    include_intercept: bool = True,
    keep_mean: bool = False,
    drop_nonfinite: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Regress out a polynomial temporal drift from trialwise patterns, feature-by-feature.

    This fits (for each feature v) an OLS model:
        X[:, v] = b0 + b1*t + b2*t^2 + ... + bK*t^K + e
    where K = `order` and t = onsets (optionally centered/scaled).

    Parameters
    ----------
    X:
        Array of shape (n_trials, n_features). Trialwise beta patterns.
    onsets:
        Array of shape (n_trials,). Trial onset times (or any monotonic time index).
        Must align with rows of X.
    order:
        Highest polynomial order to remove (K). Must be >= 0.
        - 0 removes only the intercept (demeans; if include_intercept=True).
        - 1 removes linear drift.
        - 2 removes linear + quadratic, etc.
    center_onsets:
        If True, mean-center onsets before constructing polynomial terms.
    scale_onsets:
        If True, scale onsets by their standard deviation after centering.
        Recommended when order >= 2 for numerical stability.
    include_intercept:
        If True, include an intercept column in the regression design.
        If False, removes only polynomial components without an intercept term.
        (Most users should keep this True.)
    keep_mean:
        If False (default), returns residuals (mean removed if intercept is included).
        If True, adds back the original per-feature mean after detrending, so the
        output has roughly the same mean as the input.
        Note: when include_intercept=False, keep_mean simply adds the original mean.
    drop_nonfinite:
        If True, drops trials with non-finite onsets or any non-finite values in X.
        If False, requires onsets finite and will raise if onsets contain NaN/inf.
        (X may contain NaNs; those will propagate unless you set drop_nonfinite=True.)
    eps:
        Numerical stabilizer for scaling.

    Returns
    -------
    X_detrended:
        Array of shape (n_trials, n_features) with polynomial temporal drift removed.
        If drop_nonfinite=True, trials are NOT returned to original length; instead,
        the function returns detrended data for the kept trials only.

    Notes
    -----
    - This is "voxel-by-voxel correct" in the sense that it is equivalent to running
      OLS separately for each feature; the implementation is vectorized.
    - If you want to preserve overall scaling while removing drift, use keep_mean=True.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_trials, n_features). Got {X.shape}")
    n_trials, n_feat = X.shape

    t = np.asarray(onsets, dtype=float)
    if t.ndim != 1 or t.shape[0] != n_trials:
        raise ValueError(f"onsets must be shape (n_trials,). Got {t.shape}, n_trials={n_trials}")

    if order < 0:
        raise ValueError("order must be >= 0")

    # Handle non-finite
    if drop_nonfinite:
        keep = np.isfinite(t) & np.all(np.isfinite(X), axis=1)
        Xk = X[keep]
        tk = t[keep]
    else:
        if not np.all(np.isfinite(t)):
            raise ValueError("onsets contains NaN/inf. Set drop_nonfinite=True to drop them.")
        Xk = X
        tk = t

    # Save mean to optionally add back
    mean0 = np.nanmean(Xk, axis=0)

    # Center / scale t for numerical stability
    if center_onsets:
        tk = tk - np.mean(tk)
    if scale_onsets:
        sd = np.std(tk, ddof=1)
        if sd < eps:
            # no temporal variation -> nothing to regress besides intercept
            if include_intercept:
                resid = Xk - mean0
            else:
                resid = Xk.copy()
            out = resid + (mean0 if keep_mean else 0.0)
            return out

        tk = tk / sd

    # Build design matrix: [1, t, t^2, ..., t^order] (depending on include_intercept)
    cols = []
    if include_intercept:
        cols.append(np.ones_like(tk))
    for p in range(1, order + 1):
        cols.append(tk ** p)
    D = np.column_stack(cols) if cols else np.zeros((tk.shape[0], 0), dtype=float)

    if D.shape[1] == 0:
        # nothing to remove
        out = Xk.copy()
        if include_intercept and not keep_mean:
            out = out - mean0
        return out

    # Solve OLS for all features at once: B = (D'D)^-1 D'X
    # Use lstsq for numerical robustness.
    B, *_ = np.linalg.lstsq(D, Xk, rcond=None)
    X_hat = D @ B
    resid = Xk - X_hat

    if keep_mean:
        resid = resid + mean0

    return resid


#-------------------------- run ROI analysis


def make_neural_rdv(neural, *, square_is: str = "similarity") -> np.ndarray:
    """
    Build a neural RDV (length = n_pairs, upper triangle, k=1).

    Accepts:
      - neural as (T, V) trialwise patterns -> pdist(metric="correlation")
      - neural as (T, T) square matrix:
          * square_is="similarity": treat as correlation similarity r_ij, return 1 - r_ij
          * square_is="distance": treat as already a distance matrix
      - neural as (n_pairs,) vector -> returned

    Returns:
      rdv: np.ndarray shape (T*(T-1)/2,)
    """
    arr = np.asarray(neural, dtype=float)

    if arr.ndim == 1:
        return arr.astype(float, copy=False)

    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        if square_is not in {"similarity", "distance"}:
            raise ValueError(f"square_is must be 'similarity' or 'distance', got {square_is!r}")
        D = (1.0 - arr) if square_is == "similarity" else arr
        iu, ju = np.triu_indices(D.shape[0], k=1)
        return D[iu, ju].astype(float, copy=False)

    if arr.ndim == 2:
        return pdist(arr, metric="correlation").astype(float, copy=False)

    raise ValueError(f"Unsupported neural input shape: {arr.shape}")

def fit_rsa_regression_freedman_lane(
    neural,
    design_df: pd.DataFrame,
    *,
    square_is: str = "similarity",
    n_perm: int = 500,
    seed: int = 0,
    binary_tol: float = 1e-8,
):
    """
    RSA multiple regression with Freedman–Lane-like permutations.

        z(y) ~ intercept + X

    Returns
    -------
    beta_df : pd.DataFrame
        Rows = non-intercept predictors (columns of design_df)
        Cols = beta, null_mean, null_sd, z
    r2_df : pd.DataFrame
        Single-row summary for full-model R²:
        r2, null_mean, null_sd, z, p_perm
    """

    # ----------------------------
    # neural → RDV
    # ----------------------------
    y = make_neural_rdv(neural, square_is=square_is).astype(float)
    y = y.ravel()
    L = y.size

    Xdf = design_df.copy()
    if len(Xdf) != L:
        raise ValueError("design_df length does not match neural RDV length")

    # ----------------------------
    # infer number of trials
    # ----------------------------
    T = int((1 + np.sqrt(1 + 8 * L)) / 2)
    if T * (T - 1) // 2 != L:
        raise ValueError("Invalid RDV length")

    # ----------------------------
    # helpers
    # ----------------------------
    def zscore(x):
        mu = np.mean(x)
        sd = np.std(x, ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            return np.zeros_like(x)
        return (x - mu) / sd

    def is_binary(v):
        q = np.round(v / binary_tol) if binary_tol > 0 else v
        return np.unique(q).size == 2

    def r2_score(y_true, y_hat):
        # Standard R² with intercept already in y_hat
        resid = y_true - y_hat
        sse = np.sum(resid * resid)
        y0 = y_true - np.mean(y_true)
        sst = np.sum(y0 * y0)
        if not np.isfinite(sst) or sst <= 0:
            # Degenerate case (e.g., constant y); define as 0 for stability
            return 0.0
        return 1.0 - (sse / (sst + 1e-12))

    # ----------------------------
    # clean + standardize
    # ----------------------------
    valid = np.isfinite(y)
    for c in Xdf.columns:
        valid &= np.isfinite(Xdf[c].to_numpy())

    if not valid.all():
        raise ValueError("Non-finite values detected; drop trials upstream")

    yv = zscore(y)

    Xcols = []
    names = []
    for c in Xdf.columns:
        v = Xdf[c].to_numpy()
        if is_binary(v):
            Xcols.append(v)
        else:
            Xcols.append(zscore(v))
        names.append(c)

    X = np.column_stack([np.ones(L)] + Xcols)

    # ----------------------------
    # observed fit
    # ----------------------------
    b_obs_full = np.linalg.lstsq(X, yv, rcond=None)[0]   # includes intercept
    beta_obs = b_obs_full[1:]                            # drop intercept
    yhat_obs = X @ b_obs_full
    r2_obs = r2_score(yv, yhat_obs)

    # ----------------------------
    # permutations (trial-index circular shifts)
    # ----------------------------
    iu, ju = np.triu_indices(T, k=1)
    R = np.zeros((T, T))
    R[iu, ju] = yv
    R[ju, iu] = yv

    rng = np.random.default_rng(seed)
    shifts = rng.integers(1, T, size=n_perm)

    null_betas = np.zeros((n_perm, len(names)))
    r2_null = np.zeros(n_perm, dtype=float)

    for i, k in enumerate(shifts):
        perm = (np.arange(T) + k) % T
        Rp = R[np.ix_(perm, perm)]
        y_perm = Rp[iu, ju]

        b_p_full = np.linalg.lstsq(X, y_perm, rcond=None)[0]
        null_betas[i] = b_p_full[1:]
        r2_null[i] = r2_score(y_perm, X @ b_p_full)

    # ----------------------------
    # summarize betas
    # ----------------------------
    null_mean = null_betas.mean(axis=0)
    null_sd = null_betas.std(axis=0, ddof=1)
    zvals = (beta_obs - null_mean) / (null_sd + 1e-12)

    beta_df = pd.DataFrame(
        {
            "beta": beta_obs,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "z": zvals,
        },
        index=names,
    )

    # ----------------------------
    # summarize R²
    # ----------------------------
    r2_null_mean = float(np.mean(r2_null))
    r2_null_sd = float(np.std(r2_null, ddof=1))
    r2_z = (r2_obs - r2_null_mean) / (r2_null_sd + 1e-12)

    # permutation p-value (right-tailed: "is observed >= null?")
    p_perm = (np.sum(r2_null >= r2_obs) + 1.0) / (n_perm + 1.0)

    r2_df = pd.DataFrame(
        {
            "r2": [float(r2_obs)],
            "null_mean": [r2_null_mean],
            "null_sd": [r2_null_sd],
            "z": [float(r2_z)],
            "p_perm": [float(p_perm)],
        },
        index=["full_model"],
    )

    return beta_df, r2_df


#-------------------------- plotting

def plot_rsa_results(df_results, metric="beta"):

    df_results = df_results.merge(data[['sub_id','dx']], on='sub_id', how='left')

    plt.figure(figsize=(8, 6))

    ax = sns.barplot(
        data=df_results,
        x="roi",
        y=metric,
        hue="dx",                # <-- NEW
        capsize=0,
        alpha=0.8,
        errorbar=("ci", 68),
    )

    # sns.stripplot(
    #     data=df_results,
    #     x="roi",
    #     y=metric,
    #     hue="dx",          
    #     dodge=True,             
    #     alpha=0.7,
    #     s=3,
    #     jitter=0.1,
    #     ax=ax
    # )

    ax.axhline(0, color="red", linestyle="--", linewidth=1)

    ax.set_title(f"Social location beta by dx")
    ax.set_ylabel(f"Regression {metric.capitalize()}")
    ax.set_xlabel("Region of Interest")
    plt.xticks(rotation=45, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(set(labels))], labels[:len(set(labels))], title="dx", frameon=False)
    plt.tight_layout()
    plt.show()

def roi_group_tests(
    df_rsa: pd.DataFrame,
    *,
    roi_col: str = "roi",
    y_col: str = "beta",
    dx_col: str = "dx",
    ctq_col: str = "ctq_score",
    demo_controls=("sex", "fd_mean"),
    min_n: int = 8,
    dropna: bool = True,
    hc_label: str = "HC",
    cd_label: str = "CD",
    include_F: bool = False,           # redundant when dx has 2 levels
    center_numeric: bool = True,       # makes intercept interpretable
    alpha_overall: float = 0.05,       # gating threshold for ctq interpretation
    gate_ctq_on_overall: bool = True,  # only interpret CTQ if overall mean > 0
    require_positive_means_for_dx: bool = True,  # only interpret dx if any group mean > 0
):
    """
    Per-ROI OLS with explicit HC>CD contrast:
        y ~ C(dx, Treatment(reference="CD")) + ctq_score + demo_controls...

    Adjustments for RSA interpretation:
      1) p_overall is RIGHT-TAILED for H1: mu_overall_adj > 0.
      2) For p_dx: if mean_HC <= 0 AND mean_CD <= 0, set dx stats to NaN (not interpretable).
      3) Only interpret CTQ (beta_ctq/t_ctq/p_ctq) if right-tailed p_overall is significant
         (p_overall < alpha_overall). Otherwise set CTQ stats to NaN.
    """
    d = df_rsa.copy()

    # Drop rows with missing data
    drop_cols = [roi_col, y_col, dx_col, ctq_col, *demo_controls]
    if dropna:
        d = d.dropna(subset=drop_cols).copy()

    # dx categorical
    d[dx_col] = d[dx_col].astype("category")
    levels_present = set(d[dx_col].cat.categories)
    if hc_label not in levels_present or cd_label not in levels_present:
        raise ValueError(
            f"{dx_col} must contain levels {hc_label!r} and {cd_label!r}. "
            f"Found categories: {list(d[dx_col].cat.categories)}"
        )

    # Center numeric covariates (so intercept corresponds to “at typical covariates”)
    num_to_center = [ctq_col] + [
        c for c in demo_controls
        if c in d.columns and np.issubdtype(d[c].dtype, np.number)
    ]
    if center_numeric:
        for col in num_to_center:
            x = d[col].astype(float)
            d[col] = x - float(np.nanmean(x))

    dx_term = f'C({dx_col}, Treatment(reference="{cd_label}"))'

    # Build RHS flexibly (categorical vs numeric controls)
    rhs_terms = [dx_term, ctq_col]
    for col in demo_controls:
        if col not in d.columns:
            raise ValueError(f"Missing demo control column: {col}")
        if not np.issubdtype(d[col].dtype, np.number):
            d[col] = d[col].astype("category")
            rhs_terms.append(f"C({col})")
        else:
            rhs_terms.append(col)

    formula = f"{y_col} ~ " + " + ".join(rhs_terms)

    def _right_tailed_from_ttest(t_val, p_two_sided):
        """Convert two-sided p-value from t-test to right-tailed p for H1: effect > 0."""
        if not (np.isfinite(t_val) and np.isfinite(p_two_sided)):
            return np.nan
        return (p_two_sided / 2.0) if (t_val > 0) else (1.0 - p_two_sided / 2.0)

    rows = []
    for roi, g in d.groupby(roi_col, sort=False):
        if g.shape[0] < min_n:
            continue

        means = g.groupby(dx_col)[y_col].mean()
        mean_hc = float(means.get(hc_label, np.nan))
        mean_cd = float(means.get(cd_label, np.nan))

        fit = smf.ols(formula, data=g).fit()

        dx_param = f'{dx_term}[T.{hc_label}]'

        # HC - CD
        beta_dx  = float(fit.params.get(dx_param, np.nan))
        t_dx     = float(fit.tvalues.get(dx_param, np.nan))
        p_dx     = float(fit.pvalues.get(dx_param, np.nan))

        # CTQ
        beta_ctq = float(fit.params.get(ctq_col, np.nan))
        t_ctq    = float(fit.tvalues.get(ctq_col, np.nan))
        p_ctq    = float(fit.pvalues.get(ctq_col, np.nan))

        # Intercept
        beta_0 = float(fit.params.get("Intercept", np.nan))

        # Adjusted overall mean across HC/CD at typical covariates:
        mu_overall_adj = beta_0 + 0.5 * beta_dx

        # Right-tailed test: mu_overall_adj > 0
        t_mu, p_mu_two, p_mu_right = np.nan, np.nan, np.nan
        try:
            tt = fit.t_test(f"Intercept + 0.5*{dx_param} = 0")
            t_mu = float(np.asarray(tt.tvalue).reshape(-1)[0])
            p_mu_two = float(np.asarray(tt.pvalue).reshape(-1)[0])
            p_mu_right = _right_tailed_from_ttest(t_mu, p_mu_two)
        except Exception:
            pass

        # Rule (2): if neither group > 0 on average, don't interpret dx
        if require_positive_means_for_dx:
            if not (np.isfinite(mean_hc) and np.isfinite(mean_cd)):
                beta_dx, t_dx, p_dx = np.nan, np.nan, np.nan
            elif (mean_hc <= 0) and (mean_cd <= 0):
                beta_dx, t_dx, p_dx = np.nan, np.nan, np.nan

        # Rule (3): only interpret CTQ if overall right-tailed is significant
        if gate_ctq_on_overall:
            if not (np.isfinite(p_mu_right) and (p_mu_right < alpha_overall)):
                beta_ctq, t_ctq, p_ctq = np.nan, np.nan, np.nan

        out = {
            "roi": roi,
            "N": int(g.shape[0]),
            "R2_adj": float(fit.rsquared_adj),

            "mean_HC": mean_hc,
            "mean_CD": mean_cd,

            # HC - CD (positive => HC > CD)
            "beta_dx": beta_dx,
            "t_dx": t_dx,
            "p_dx": p_dx,

            # CTQ slope (gated if requested)
            "beta_ctq": beta_ctq,
            "t_ctq": t_ctq,
            "p_ctq": p_ctq,

            # adjusted overall mean across groups
            "mu_overall": float(mu_overall_adj),
            "t_overall": t_mu,
            "p_overall": p_mu_right,      # RIGHT-tailed p-value for mu_overall > 0
            "p_overall_2s": p_mu_two,     # optional: keep the two-sided version for transparency
        }

        if include_F:
            out["F_dx"] = float(t_dx**2) if np.isfinite(t_dx) else np.nan

        rows.append(out)

    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(["p_overall", "p_dx", "p_ctq"], na_position="last")
    return res

def plot_sig_rois_from_results(
    results_df,
    atlas,
    *,
    p_col: str,
    alpha: float = 0.05,
    roi_col: str = "roi",
    score: str = "-log10p",          # {"-log10p","1-p","binary"}
    display_mode: str = "ortho",
    cmap: str | None = "viridis",
    cut_coords=None,
    title: str | None = None,
):
    """
    Plot all ROIs with results_df[p_col] < alpha, colored by significance.

    Creates a single 3D overlay image:
      voxel value = f(p) within each significant ROI, else 0.
    """

    if roi_col not in results_df.columns:
        raise ValueError(f"results_df missing roi_col={roi_col!r}")
    if p_col not in results_df.columns:
        raise ValueError(f"results_df missing p_col={p_col!r}")

    # if atlas is a string, load it
    if isinstance(atlas, str):
        atlas = load_pickle(atlas)
    rois = atlas["rois"]
    atlas_img = atlas["image"]

    # Build label -> atlas_code mapping
    if isinstance(rois, dict):
        code_to_label = {int(k): str(v) for k, v in rois.items()}
        label_to_code = {v: k for k, v in code_to_label.items()}
    else:
        labels = [str(x) for x in list(rois)]
        label_to_code = {lab: (i + 1) for i, lab in enumerate(labels)}  # implicit

    # Filter significant rows
    d = results_df[[roi_col, p_col]].copy()
    d[p_col] = np.asarray(d[p_col], float)
    d = d[np.isfinite(d[p_col])]
    d = d[d[p_col] < alpha].sort_values(p_col, ascending=True)

    # warn + break/exit early if nothing survives
    if d.empty:
        print(f"Warning: No ROIs with {p_col} < {alpha}. Nothing to plot.")
        return None, None

    # Create overlay volume (same grid as atlas)
    atlas_data = atlas_img.get_fdata().astype(int)
    overlay = np.zeros(atlas_data.shape, dtype=float)

    # Fill overlay ROI-by-ROI
    missing = []
    for _, row in d.iterrows():
        roi_label = str(row[roi_col])
        p = float(row[p_col])

        if roi_label not in label_to_code:
            missing.append(roi_label)
            continue

        code = int(label_to_code[roi_label])
        mask = (atlas_data == code)

        if score == "-log10p":
            val = -np.log10(max(p, 1e-300))
        elif score == "1-p":
            val = 1.0 - p
        elif score == "binary":
            val = 1.0
        else:
            raise ValueError("score must be one of {'-log10p','1-p','binary'}")

        overlay[mask] = val

    if missing:
        print(
            f"Warning: {len(missing)} ROI labels not found in atlas "
            f"(showing up to 20): {missing[:20]}"
        )

    overlay_img = nimg.new_img_like(atlas_img, overlay, copy_header=True)

    if title is None:
        title = f"Significant ROIs: {p_col} < {alpha} (n={len(d)})"

    plotting.plot_stat_map(
        overlay_img,
        title=title,
        display_mode=display_mode,
        cut_coords=cut_coords,
        threshold=0,
        draw_cross=False,
        cmap=cmap,
    )
    plotting.show()

    return d, overlay_img

def plot_atlas_roi_from_pkl(atlas, roi_label, *, display_mode="ortho", cut_coords=None):
    """
    Supports atlas["rois"] as either:
      - dict[int -> str] : atlas codes map to label names (your case)
      - list/tuple[str]  : labels by index (then we assume code = index+1)
    """
    atlas_img = atlas["image"]
    rois = atlas["rois"]

    # --- build label -> atlas_code mapping ---
    if isinstance(rois, dict):
        # dict: code -> label
        code_to_label = {int(k): str(v) for k, v in rois.items()}
        label_to_code = {v: k for k, v in code_to_label.items()}

        if roi_label not in label_to_code:
            matches = [lab for lab in label_to_code if roi_label.lower() in lab.lower()]
            raise ValueError(f"ROI label not found: {roi_label!r}. Close matches: {matches[:20]}")

        atlas_code = int(label_to_code[roi_label])   # e.g., 109 for Left Hippocampus

    else:
        # list-like: label at position i corresponds to code i+1
        labels = [str(x) for x in list(rois)]
        if roi_label not in labels:
            matches = [lab for lab in labels if roi_label.lower() in lab.lower()]
            raise ValueError(f"ROI label not found: {roi_label!r}. Close matches: {matches[:20]}")

        i = labels.index(roi_label)
        atlas_code = i + 1

    # --- build mask using atlas_code ---
    data = atlas_img.get_fdata().astype(int)
    roi_mask = (data == atlas_code).astype(np.uint8)
    roi_img = nimg.new_img_like(atlas_img, roi_mask, copy_header=True)

    title = f"{roi_label} (atlas_code={atlas_code})"
    plotting.plot_roi(roi_img, title=title, display_mode=display_mode, cut_coords=cut_coords)
    plotting.show()

    return roi_img, atlas_code


#-------------------------- run 2nd level analysis

def threshold_neglog_img(neglog_in, alpha=0.05, cluster_threshold=0,
                         to_voxels=True, two_sided=True):
    # normalize input to Nifti
    neglog_img = image.load_img(neglog_in['logp_max_mass'] if isinstance(neglog_in, dict) else neglog_in)

    thr = float(-np.log10(alpha))
    mask_expr = f"(np.abs(img) >= {thr})" if two_sided else f"(img >= {thr})"
    mask    = image.math_img(mask_expr, img=neglog_img)
    thr_img = image.math_img("img * m", img=neglog_img, m=mask)

    # cluster table (mm^3)
    table = get_clusters_table(thr_img, stat_threshold=thr, cluster_threshold=cluster_threshold)

    # vectorized mm^3 → voxels
    if to_voxels and len(table) > 0 and 'Cluster Size (mm3)' in table.columns:
        vx_mm3 = float(abs(np.linalg.det(neglog_img.affine[:3, :3])))
        # coerce to numeric; non-numeric → NaN, then divide in one shot
        mm3 = pd.to_numeric(table['Cluster Size (mm3)'], errors='coerce')
        table['Cluster Size (voxels)'] = mm3 / vx_mm3
        table.drop(columns=['Cluster Size (mm3)'], inplace=True)

    return thr_img, table

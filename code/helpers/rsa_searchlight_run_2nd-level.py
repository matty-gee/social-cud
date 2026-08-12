from utils_fmri import *
from utils import data

from nilearn.glm.second_level import SecondLevelModel
from nilearn import image
from nilearn.input_data import NiftiMasker

# -------------------------- helpers

def find_sl_imgs(sl_dir, sl_model):
    incl_subs = [str(sid) for sid in data['sub_id'].tolist()]
    sl_niis = glob.glob(f'{sl_dir}/*{sl_model}*')
    sl_niis = [s for s in sl_niis  if s.split('/')[-1].split('_')[0].removeprefix("sub-") in incl_subs]
    sl_dict = {'sub_ids':[], 'imgs':[]}
    for sl_nii in sl_niis:
        sub_id = sl_nii.split('/')[-1].split('_')[0].removeprefix("sub-")
        sl_dict['sub_ids'].append(sub_id)        
        sl_dict['imgs'].append(sl_nii)
    print(f'Found {len(sl_dict["sub_ids"])} included searchlight images for n={len(incl_subs)}')
    return sl_dict

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def build_design_matrix(
    sub_ids,
    imgs,
    *,
    covariates=None,
    data=None,
    zscore=False,
    zscore_within_dx=True,
):
    """
    Build a second-level design matrix.

    Parameters
    ----------
    sub_ids : iterable
        Subject identifiers.
    imgs : iterable
        Subject labels passed to nilearn's make_second_level_design_matrix
        (same order as sub_ids).
    covariates : list of str or None, optional
        Subject-level covariates to include.
        Special cases:
          - "reaction_time" -> uses the RT-difference helper:
              |mean RT_affil − mean RT_power| from load_behavior(...)
          - "sex" -> mapped from data['sex'] via: M -> 0, F -> 1
        All other names are interpreted as column names in `data`, looked up per
        sub_id.
        If None or empty, only dx and ctq_score (plus intercept) are included.
    data : pandas.DataFrame or None, optional
        DataFrame with at least columns ["sub_id", "dx", "ctq_score"] +
        covariates (except "reaction_time", which does not require a column).
        If None, we fall back to a global `data` variable when non-RT
        covariates are requested.
    zscore : bool, optional
        If True, z-score numeric covariates (excluding dx and sex).
    zscore_within_dx : bool, optional
        If True and zscore is True and both dx groups are present, z-score
        numeric covariates within dx group (0 vs 1). If False, z-score across
        all subjects.

    Returns
    -------
    design_matrix : pandas.DataFrame
        Second-level design matrix aligned to `imgs`, with columns:
        intercept, dx (0=HC, 1=CD), ctq_score, sex (if included), and any
        requested covariates.
    """

    # -------- helpers --------

    if covariates is None:
        covariates = []  # no extra covariates by default

    if data is None and any(
        cov != "reaction_time" for cov in covariates + ["dx", "ctq_score"]
    ):
        # fall back to global `data` if user didn't pass one explicitly
        try:
            data = globals()["data"]
        except KeyError:
            raise ValueError(
                "Non-RT covariates or dx/ctq_score requested but neither `data` "
                "argument nor global `data` is available."
            )

    def rt_diff_for(sub_id):
        """|mean RT_affil − mean RT_power| from trialwise behavior."""
        behav = load_behavior(sub_id, neutrals=False)
        behav = behav.loc[behav["responded"]]
        m = behav.groupby("dimension")["reaction_time"].mean()
        return abs(m.get("affil", np.nan) - m.get("power", np.nan))

    def _row_for_sub(sub_id):
        """
        Return the row(s) in `data` for this subject, matching on numeric sub_id
        when possible.
        """
        if data is None:
            return data  # None

        # Prefer numeric match (data['sub_id'] is numeric in your setup)
        try:
            key_num = int(sub_id)
            sel = data.loc[data["sub_id"] == key_num]
            if not sel.empty:
                return sel
        except (TypeError, ValueError):
            pass

        # Fallback: direct equality (in case data['sub_id'] is not numeric)
        sel = data.loc[data["sub_id"] == sub_id]
        return sel

    def subj_value_for(sub_id, col):
        """Generic subject-level lookup from `data`."""
        if data is None:
            return np.nan
        sel = _row_for_sub(sub_id)
        if sel is None or sel.empty:
            return np.nan
        val = sel[col].iloc[0]
        # Try to cast to float; if it fails (e.g., non-numeric), return as-is
        try:
            return float(val)
        except (TypeError, ValueError):
            return val

    def dx_for(sub_id):
        """
        Map dx string to numeric:
        HC -> 0, CD -> 1, else NaN.
        """
        if data is None:
            return np.nan
        sel = _row_for_sub(sub_id)
        if sel is None or sel.empty or "dx" not in sel.columns:
            return np.nan
        val = sel["dx"].iloc[0]
        if val == "HC":
            return 0.0
        elif val == "CD":
            return 1.0
        else:
            return np.nan

    def sex_for(sub_id):
        """
        Map sex string to numeric:
        M -> 0, F -> 1, else NaN.
        """
        if data is None:
            return np.nan
        sel = _row_for_sub(sub_id)
        if sel is None or sel.empty or "sex" not in sel.columns:
            return np.nan
        val = sel["sex"].iloc[0]
        if val == "M":
            return 0.0
        elif val == "F":
            return 1.0
        else:
            return np.nan

    def ctq_for(sub_id):
        return subj_value_for(sub_id, "ctq_score")

    # -------- basic checks --------

    sub_ids = list(sub_ids)
    imgs = list(imgs)
    if len(sub_ids) != len(imgs):
        raise ValueError("sub_ids and imgs must have the same length")

    # ----------------------
    # 1) Build confounds
    # ----------------------

    rows = []
    for sid, img in zip(sub_ids, imgs):
        row = {
            "subject_label": img,
            "sub_id": sid,
            # Always include dx and ctq_score from `data`
            "dx": dx_for(sid),
            "ctq_score": ctq_for(sid),
        }

        for cov in covariates:
            if cov == "reaction_time":
                # special helper: |mean RT_affil − mean RT_power|
                row[cov] = rt_diff_for(sid)
            elif cov == "sex":
                # special mapping: M -> 0, F -> 1
                row["sex"] = sex_for(sid)
            else:
                # generic subject-level covariate from `data`
                row[cov] = subj_value_for(sid, cov)

        rows.append(row)

    confound_df = pd.DataFrame(rows)

    # Ensure dx, ctq_score, sex are numeric where present
    for col in ["dx", "ctq_score", "sex"]:
        if col in confound_df.columns:
            confound_df[col] = pd.to_numeric(confound_df[col], errors="coerce")

    # ----------------------
    # 2) Optional z-scoring of numeric covariates
    # ----------------------
    if zscore:
        # Numeric columns, excluding labels and binary codes dx/sex
        numeric_cols = []
        for col in confound_df.columns:
            if col in ("subject_label", "sub_id", "dx", "sex"):
                continue
            if np.issubdtype(confound_df[col].dtype, np.number):
                numeric_cols.append(col)

        if numeric_cols:
            if zscore_within_dx and "dx" in confound_df.columns and confound_df["dx"].nunique() > 1:
                # z-score within dx group
                for g, idx in confound_df.groupby("dx").groups.items():
                    for col in numeric_cols:
                        vals = confound_df.loc[idx, col]
                        mu = vals.mean()
                        sd = vals.std(ddof=1)
                        if np.isfinite(sd) and sd > 0:
                            confound_df.loc[idx, col] = (vals - mu) / sd
                        else:
                            # fallback: mean-center only
                            confound_df.loc[idx, col] = vals - mu
            else:
                # global z-scoring across all subjects
                for col in numeric_cols:
                    vals = confound_df[col]
                    mu = vals.mean()
                    sd = vals.std(ddof=1)
                    if np.isfinite(sd) and sd > 0:
                        confound_df[col] = (vals - mu) / sd
                    else:
                        confound_df[col] = vals - mu

    # ----------------------
    # 3) Design-matrix inputs
    # ----------------------

    cov_cols = [c for c in confound_df.columns if c not in ("subject_label", "sub_id")]
    dm_input = confound_df[["subject_label"] + cov_cols]

    # ----------------------
    # 4) Build the second-level design matrix
    # ----------------------

    design_matrix = make_second_level_design_matrix(
        subjects_label=imgs,
        confounds=dm_input,
    )

    # ----------------------
    # 5) Return a clean, ordered DM
    #    (intercept first, then dx, ctq_score, sex, then others)
    # ----------------------

    out_cols = ["intercept"]
    for base in ["dx", "ctq_score", "sex"]:
        if base in design_matrix.columns:
            out_cols.append(base)

    remaining = [c for c in design_matrix.columns if c not in out_cols]
    out_cols.extend(remaining)

    return design_matrix[out_cols]

def compute_permutation_ttest(fnames, mask_img=None, 
                              fwhm=None, 
                              design_matrix=None, 
                              second_level_contrast=None, 
                              confounds=None,
                              model_intercept=True, 
                              two_sided=False, 
                              threshold=None, 
                              n_perm=10000, 
                              tfce=False):

    ''' 
        Compute a permutation t-test on a list of first level contrast images

        Arguments
        ---------
        fnames: list of str
            list of paths to first level contrast images
        confound_df: pd.DataFrame
            dataframe with confounds, needs “subject_label” column
        mask_img: str
            path to mask image
        fwhm: float
            smoothing kernel in mm
        two_sided: bool
            whether to use two-sided test
        model_intercept: bool
            whether to model intercept
        n_perm: int
            number of permutations
        threshold: float
            p-scale cluster forming threshold
        tfce: bool
            whether to use threshold-free cluster enhancement

        Returns
        -------
        if threshold is None: negative logarithm of the voxel-level FWER-corrected p-values
        if threshold is not None: dictionary with keys (see: https://nilearn.github.io/dev/modules/generated/nilearn.glm.second_level.non_parametric_inference.html#nilearn.glm.second_level.non_parametric_inference)
    '''
    
    # https://nilearn.github.io/dev/modules/generated/nilearn.glm.second_level.non_parametric_inference.html

    print(f'Running nonparametric 1-sample t-test, n={len(fnames)}')

    if design_matrix is None: # assumes 1 sample t-test
        design_matrix = pd.DataFrame([1] * len(fnames), columns=['intercept']) 
    
    # produces a negative log pvalue image or an output dictionary if tfce=False and threshold is not None
    return non_parametric_inference(fnames, 
                                    design_matrix=design_matrix, 
                                    confounds=confounds, # if used, needs “subject_label” column
                                    model_intercept=model_intercept, 
                                    first_level_contrast=None, 
                                    second_level_contrast=second_level_contrast, 
                                    mask=mask_img, 
                                    smoothing_fwhm=fwhm, 
                                    n_perm=n_perm, # number of 0s determines precision of the p-value
                                    two_sided_test=two_sided, 
                                    threshold=threshold, # p-scale cluster forming threshold
                                    tfce=tfce, # as described in orig. paper
                                    random_state=2022, 
                                    n_jobs=-1, 
                                    verbose=1)


def contrast_vec(design_matrix: pd.DataFrame, name: str) -> np.ndarray:
    cols = list(design_matrix.columns)
    if name not in cols:
        raise ValueError(f"Contrast '{name}' not in design_matrix columns: {cols}")
    c = np.zeros(len(cols), dtype=float)
    c[cols.index(name)] = 1.0
    return c

def contrast_hc_gt_cd(design_matrix: pd.DataFrame, dx_col: str = "dx") -> np.ndarray:
    """
    With your coding dx: HC=0, CD=1, HC>CD corresponds to -dx.
    """
    cols = list(design_matrix.columns)
    if dx_col not in cols:
        raise ValueError(f"'{dx_col}' not in design matrix columns: {cols}")
    c = np.zeros(len(cols), dtype=float)
    c[cols.index(dx_col)] = -1.0
    return c

def extract_avg_roi_ests_from_imgs(sub_ids, imgs, mask_dict, *, standardize=False):
    """
    Mean value within each ROI mask from each subject image.
    Uses one fitted masker per ROI (faster + simple).
    """
    sub_ids = list(sub_ids)
    imgs = list(imgs)
    if len(sub_ids) != len(imgs):
        raise ValueError("sub_ids and imgs must have same length")

    ref_img = nib.load(imgs[0]) if isinstance(imgs[0], str) else imgs[0]

    roi_names = list(mask_dict.keys())
    maskers = {
        roi: NiftiMasker(mask_img=mask_dict[roi], standardize=standardize).fit(ref_img)
        for roi in roi_names
    }

    rows = []
    for sid, img in zip(sub_ids, imgs):
        row = {"sub_id": sid}
        for roi in roi_names:
            X = maskers[roi].transform(img)  # (1, n_vox)
            row[roi] = float(np.nanmean(X)) if X.size else np.nan
        rows.append(row)

    return pd.DataFrame(rows)

# -------------------------- run the analysis

overwrite     = True      
overwrite_roi = False     

fwhm        = 8
tfce        = False
n_perm      = 1000

results_dir = "../analyses/lsa/searchlights/results"
param_dir   = ensure_dir(f"{results_dir}/parametric")
perm_dir    = ensure_dir(f"{results_dir}/permutation")

covariates  = ["sex", "fd_mean", 'memory_mean']
model_dict  = {"location-within_time-sq": covariates}
# trajsim-ew_temp-2": covariates}
# "location_time-sq": covariates, 
# define masks
mask_dir = "../masks/ROIs"
ref_img  = nib.load('../masks/example_image_for_resampling.nii.gz') 
mask_dict = {
    "HPC": nib.load(f"{mask_dir}/HPC-bilat_harvardoxford_maxprob-thr50_1mm.nii.gz"),
    "GM":  nib.load(f"{mask_dir}/GM.nii.gz"),
}
mask_dict = {
    name: image.resample_to_img(mask, ref_img, interpolation="nearest")
    for name, mask in mask_dict.items()
}

# -------------------------- per-model loop --------------------------

for sl_model, covs in model_dict.items():

    sl_dir    = f"../analyses/lsa/searchlights/{sl_model}/"
    sl_dict   = find_sl_imgs(sl_dir, f"{sl_model}_ball")
    sub_ids   = sl_dict["sub_ids"]
    imgs      = sl_dict["imgs"]
    sample_n  = len(sub_ids)

    # ------------------ ROI means from first-level images

    out_csv = f"{results_dir}/roi/{sl_model}_roi_mean_ests.csv"
    if os.path.exists(out_csv) and not overwrite_roi:
        print(f"Found existing ROI estimates CSV: {out_csv}")
        est_df = pd.read_csv(out_csv)
    else:
        est_df = extract_avg_roi_ests_from_imgs(
            sub_ids,
            imgs,
            mask_dict,
            standardize=False,
        )
        est_df["sub_id"] = pd.to_numeric(est_df["sub_id"], errors="coerce")
        est_df = est_df.merge(data, on="sub_id", how="left")
        est_df.to_csv(out_csv, index=False)
        print(f"[roi] saved: {out_csv}")

    # ------------------ build design matrix

    design_matrix = build_design_matrix(
        sub_ids,
        imgs,
        covariates=covs,
        data=data,
        zscore=True,
        zscore_within_dx=False,
    )

    # ------------------ PARAMETRIC: unthresholded z-maps 

    model = SecondLevelModel(
        smoothing_fwhm=fwhm,
        minimize_memory=False,
        target_affine=affine,
        n_jobs=-1,
    )
    model.fit(imgs, design_matrix=design_matrix)

    param_contrasts = {
        "hc":  contrast_vec(design_matrix, "intercept"), 
        "dx":  contrast_hc_gt_cd(design_matrix, "dx"),
        "ctq": contrast_vec(design_matrix, "ctq_score"),
        "memory": contrast_vec(design_matrix, "memory_mean"),
    }

    for cname, cvec in param_contrasts.items():
        out_fname = f"{param_dir}/{sl_model}_{cname}_{fwhm}fwhm_n{sample_n}_zmap.nii.gz"
        if (not overwrite) and os.path.exists(out_fname):
            continue
        z_map = model.compute_contrast(cvec, output_type="z_score")
        z_map.to_filename(out_fname)
        print(f"[parametric] saved: {out_fname}")

    # ------------------ NONPARAMETRIC: FWER-corrected unthresholded -log10(p) 

    perm_contrasts = {
        "hc":  "intercept",
        "dx":  contrast_hc_gt_cd(design_matrix, "dx"),
        "ctq": "ctq_score",
        "memory": "memory_mean",
    }

    for mask_name, mask_img in mask_dict.items():
        for cname, contrast in perm_contrasts.items():
            out_fname = f"{perm_dir}/{sl_model}_{mask_name}_{cname}_{fwhm}fwhm_n{sample_n}_neglogp.nii.gz"
            if (not overwrite) and os.path.exists(out_fname):
                continue
            ttest_out = compute_permutation_ttest(
                imgs,
                mask_img=mask_img,
                n_perm=n_perm,
                fwhm=fwhm,
                design_matrix=design_matrix,
                second_level_contrast=contrast,  # string or numeric vector
                two_sided=False,
                model_intercept=True,
                threshold=None,  # <- unthresholded output
                tfce=tfce,
            )
            ttest_out.to_filename(out_fname)
            print(f"[perm] saved: {out_fname}")

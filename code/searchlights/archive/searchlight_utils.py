import os, glob
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import itertools
import scipy
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import rankdata
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
import brainiak.searchlight.searchlight
import brainiak.searchlight.searchlight as bk_sl
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


snt_df = pd.read_excel("social-navigation-task.xlsx")
snt_df = snt_df[np.isfinite(snt_df["trial_num"])]
snt_df.sort_values(by="onset", inplace=True)
decision_trials = snt_df[snt_df["slide_type"] == "Decision"].copy()


########################################################################
# HELPERS 
########################################################################


def flatten_upper_tri(mat):
    """ go from symmetrical matrix to vectorized/flattened upper triangle """
    return mat[np.triu_indices(len(mat), k=1)]

# behavioral helpers
def load_behavior(sub_id, neutrals=True):

    # helper for getting behavior dataframe, locally or on Minerva
    df = pd.read_excel(f"../data/preprocessed/behavior/{sub_id if 'sub-' in str(sub_id) else 'sub-' + str(sub_id)}.xlsx")
    if not neutrals:
        df = df[df['character_role_num'] != 6].reset_index(drop=True)
    return df

def remove_neutral_trials(arr):
    '''remove the neutral trials from array: trials 15,16,36
    '''
    assert arr.shape[0] == 63, 'input array must have 63 trials'
    return np.delete(arr, np.array([14,15,35]), axis=0) 

def organize_by_character(data, neutrals=False):

    if data.shape[0] == 63:
        trialwise_char_roles = decision_trials['character_role_num'].values
        if neutrals:
            n_chars = 6
        else: 
            n_chars = 5
    elif data.shape[0] == 60:
        trialwise_char_roles = remove_neutral_trials(decision_trials['character_role_num'].values)
        n_chars = 5
    if isinstance(data, np.ndarray):
        if data.ndim == 1: 
            return [data[trialwise_char_roles == c] for c in range(1,n_chars+1)]
        else: 
            return [data[trialwise_char_roles == c,:] for c in range(1,n_chars+1)]
    elif isinstance(data, pd.DataFrame):
        return [data[trialwise_char_roles == c] for c in range(1, n_chars+1)]
    
# for covariate RDVs
def time_poly_basis(onsets, degrees=(1, 2), zscore_cols=True, eps=1e-12):
    """
    Return array with columns for |Δt|^d where higher orders are residualized
    on all lower orders (with intercept). Optionally z-score columns.
    Shape: [n_pairs, len(degrees)].
    """
    def orth_residualize(x, *others):
        n = len(x)
        if not others:
            return x - x.mean()
        X = np.column_stack([np.ones(n), *others])
        beta, *_ = np.linalg.lstsq(X, x, rcond=None)
        return x - X @ beta

    def zscore(x):
        x = np.asarray(x, float)
        return (x - x.mean()) / (x.std(ddof=1) + eps)

    dt = pdist(np.asarray(onsets, float).reshape(-1, 1), metric='euclidean')
    degs = tuple(sorted(degrees))

    cols = []
    for j, d in enumerate(degs):
        x = dt ** d
        if j > 0:  # make 'unique' by removing projections on lower orders
            x = orth_residualize(x, *cols)
        if zscore_cols:
            x = zscore(x)
        cols.append(x)

    return np.column_stack(cols)

def per_character_rdvs(char_id_vec):
    """
    Return dict {character_value: rdv_vector} where each RDV is 1 for
    within-character pairs (both endpoints == that character), else 0.

    The RDVs are in SciPy's condensed order (same as pdist).
    """
    char_id_vec = np.asarray(char_id_vec)
    chars = np.unique(char_id_vec)
    out = {}
    for c in chars:
        mask = (char_id_vec == c).astype(float)  # length T
        M = np.outer(mask, mask)                 # T x T, 1 only if both endpoints are c
        np.fill_diagonal(M, 0.0)                 # hollow
        out[c] = squareform(M, checks=False).astype(float)
    return out

def create_behavioral_rdvs(behavior,
                           zscore_continuous=True,
                           remove_neutrals=True):
    """
    Behavior-only RDVs with:
      • Orthogonalized time polynomials via time_poly_basis (deg 1,2,3).
      • Character-specific same/diff RDVs via per_character_rdvs (character_01, ...).
    """

    # optional neutral removal
    if remove_neutrals:
        keep = (behavior['character_role_num'].values != 6)
        behavior = behavior.loc[keep].reset_index(drop=True)

    # columns
    xys    = behavior[['affil_coord', 'power_coord']].values
    rts    = behavior['reaction_time'].values
    dims   = (behavior['dimension'].values == 'affil').astype(int)
    chars  = behavior['character_role_num'].values
    fam    = behavior['character_decision_num'].values
    onsets = behavior['onset'].values
    scenes = behavior['scene_num'].values

    # base RDVs
    rdvs = {
        'location_euc'  : pdist(xys, metric='euclidean'),
        'reaction_time' : pdist(rts.reshape(-1, 1), metric='euclidean'),
        'dimension'     : pdist(dims.reshape(-1, 1), metric='hamming').astype(int),   # same/diff
        'character'     : pdist(chars.reshape(-1, 1), metric='hamming').astype(int),  # same/diff (pooled)
        'scene'         : pdist(scenes.reshape(-1, 1), metric='hamming').astype(int), # same/diff
    }

    # familiarity (linear; add quadratic later if desired)
    rdvs['familiarity_linear'] = pdist(fam.reshape(-1, 1), metric='euclidean')

    # ---- time polynomials (orthogonalized; already z-scored inside helper if zscore_cols=True)
    T = time_poly_basis(onsets, degrees=(1, 2, 3), zscore_cols=True)  # shape [N_pairs, 3]
    rdvs['time_linear']    = T[:, 0]  # |Δt| (z-scored)
    rdvs['time_quadratic'] = T[:, 1]  # curvature beyond linear (z-scored)
    rdvs['time_cubic']     = T[:, 2]  # cubic beyond linear+quadratic (z-scored)

    # ---- character-specific same/diff RDVs (per-character masks)
    per_char = per_character_rdvs(chars)  # {char_value: rdv (float 0/1)}
    for c, v in per_char.items():
        rdvs[f'character_{int(c):02d}'] = v.astype(int)

    # ---- apply optional z-scoring to continuous RDVs
    per_char_keys = {f'character_{int(c):02d}' for c in np.unique(chars)}
    categorical   = {'dimension', 'character', 'scene'} | per_char_keys
    continuous    = set(rdvs.keys()) - categorical
    skip_zscore   = {'time_linear', 'time_quadratic', 'time_cubic'} # already z-scored
    for k in continuous:
        v = np.asarray(rdvs[k], float)
        if zscore_continuous and (k not in skip_zscore):
            v = (v - v.mean()) / (v.std() + 1e-12)
        rdvs[k] = v

    return rdvs

# fmri image helpers
def get_nifti_info(nifti):
    """Return (dimensions, voxel size [xyz], affine) of a NIfTI without forcing a full data load."""
    if isinstance(nifti, (str, Path)):
        nifti = nib.load(str(nifti))
    elif not isinstance(nifti, nib.spatialimages.SpatialImage):
        raise TypeError("get_nifti_info: expected path or NIfTI image")

    dims = nifti.shape
    vox_size = nifti.header.get_zooms()[:3]
    affine_matrix = nifti.affine
    return dims, vox_size, affine_matrix

def save_as_nifti(brain_data, output_name, affine_mat, vox_size):
    """
    Save a brain matrix as a NIfTI file.

    brain_data : np.ndarray (numeric or object with None)
    affine_mat : (4,4) array
    vox_size   : (3,) tuple/list of voxel sizes
    """
    def _object_to_float_array(arr):
        """Convert BrainIAK object arrays (with None) to float64 with NaN; numeric arrays pass through."""
        arr = np.asarray(arr)
        if arr.dtype == object:
            vfunc = np.vectorize(lambda v: np.nan if v is None else float(v), otypes=[np.float64])
            return vfunc(arr)
        return arr.astype(np.float64, copy=False)

    arr = _object_to_float_array(brain_data)
    # Convert to a precision compatible elsewhere; replace NaNs with zero (your original behavior)
    arr[np.isnan(arr)] = 0.0
    brain_nii = nib.Nifti1Image(arr.astype(np.float64, copy=False), affine_mat)

    hdr = brain_nii.header
    # Never write a 0 TR; if you don't know TR, write 1.0 (safer than 0)
    if arr.ndim == 4:
        hdr.set_zooms((float(vox_size[0]), float(vox_size[1]), float(vox_size[2]), 1.0))
    else:
        hdr.set_zooms((float(vox_size[0]), float(vox_size[1]), float(vox_size[2])))

    nib.save(brain_nii, str(output_name))
    print(f"[save_as_nifti] Saved: {output_name} | shape={arr.shape}, dtype={arr.dtype}")


########################################################################
# SEARCHLIGHTS
# add the running mean and other analyses of interest...
########################################################################

analysis_dir  = "/sc/arion/projects/OlfMem/mgs/analyses"

#------------------- general helpers

class Searchlight:
    """
    Convenience wrapper around brainiak.searchlight.searchlight.Searchlight

    Public API preserved:
      - __init__(sl_f, shape='ball', radius=3, min_prop=0.10, num_sls=10)
      - prepare(func_imgs, func_masks=None, roi_masks=None, bcvar=None)
      - run(save=True, out_prefix='')

    Notes / Fixes:
      * Ensures boolean 3D mask is passed to BrainIAK.
      * Correctly handles single-subject mask (no list leakage).
      * Fixes undefined `self.subject_mask` -> `self.func_mask`.
      * Uses `pool_size=self.num_sls`; does not misuse `max_blk_edge`.
      * Prints a mask-only min_prop coverage pre-check (what BrainIAK gates on).
      * Optionally saves combined ROI/functional mask with same out_prefix.
    """
    def __init__(self, sl_f, shape='ball', radius=3, min_prop=0.10, num_sls=10):
        assert shape in {'cube', 'ball', 'diamond'}, "shape must be one of {'cube','ball','diamond'}"
        assert radius >= 0 and int(radius) == radius, "radius must be a non-negative integer"
        assert 0 <= float(min_prop) <= 1.0, "min_prop must be in [0, 1]"
        assert isinstance(num_sls, int) and num_sls >= 1, "num_sls must be a positive integer"

        self.sl_f = sl_f
        self.radius = int(radius)
        self.min_prop = float(min_prop)
        self.num_sls = int(num_sls)
        self.shape_name = shape

        sl_shapes = {'cube': bk_sl.Cube, 'ball': bk_sl.Ball, 'diamond': bk_sl.Diamond}
        self.shape = sl_shapes[shape]  # CLASS, BrainIAK instantiates internally

        # placeholders filled in prepare()
        self.func_imgs = None
        self.func_masks = None
        self.roi_masks = None
        self.bcvar = None
        self.brain_data = None
        self.brain_mask = None
        self.affine_matrix = None
        self.vox_size = None

    # ---------- helpers ----------
    @staticmethod
    def _load_to_array(fname_or_img):
        img = nib.load(str(fname_or_img)) if isinstance(fname_or_img, (str, Path)) else fname_or_img
        if not isinstance(img, nib.spatialimages.SpatialImage):
            raise TypeError("Expected path or NIfTI image")
        return img.get_fdata(dtype=np.float32)

    @staticmethod
    def _ensure_bool3d(niimg):
        if not isinstance(niimg, nib.spatialimages.SpatialImage):
            niimg = nib.load(str(niimg))
        arr = niimg.get_fdata()
        return (arr > 0).astype(bool)

    @staticmethod
    def _prop_from_mask(mask_bool, shape='ball', radius=3):
        """Compute per-voxel proportion of active mask within neighborhood (mask-only, BrainIAK gate)."""
        r = int(radius)
        xs, ys, zs = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
        if shape == 'ball':
            fp = (xs*xs + ys*ys + zs*zs) <= r*r
        elif shape == 'diamond':
            fp = (np.abs(xs) + np.abs(ys) + np.abs(zs)) <= r
        else:  # cube
            fp = np.ones((2*r+1, 2*r+1, 2*r+1), dtype=bool)
        fp = fp.astype(np.float32)
        S = fp.sum()
        counts = convolve(mask_bool.astype(np.float32), fp, mode="constant", cval=0.0)
        return (counts / S).astype(np.float32), int(S)

    @staticmethod
    def _object_to_float_array(arr):
        """Convert BrainIAK object arrays (with None) to float64 with NaN; numeric arrays pass through."""
        arr = np.asarray(arr)
        if arr.dtype == object:
            vfunc = np.vectorize(lambda v: np.nan if v is None else float(v), otypes=[np.float64])
            return vfunc(arr)
        return arr.astype(np.float64, copy=False)

    def prepare(self, func_imgs, func_masks=None, roi_masks=None, bcvar=None):
        """Load data, combine masks, and compute mask coverage diagnostics."""
        self.func_imgs  = func_imgs
        self.func_masks = func_masks
        self.roi_masks  = roi_masks
        self.bcvar      = bcvar

        # ---- load functional data ----
        if isinstance(func_imgs, (str, Path)):
            print("[prepare] Loading single subject's functional image")
            func_list = [func_imgs]
        elif isinstance(func_imgs, list):
            print("[prepare] Loading multiple subjects' functional images")
            func_list = func_imgs
        else:
            raise TypeError("func_imgs must be a path or a list of paths")

        imgs = [nib.load(str(f)) for f in func_list]
        self.brain_data = [img.get_fdata(dtype=np.float32) for img in imgs]

        # consistency checks
        shapes_3d = [arr.shape[:3] for arr in self.brain_data]
        assert len(set(shapes_3d)) == 1, f"All subjects must have same spatial shape; got {set(shapes_3d)}"
        n_subs = len(func_list)
        _, self.vox_size, self.affine_matrix = get_nifti_info(imgs[0])
        print(f"[prepare] n_subjects={n_subs} | data_shape={self.brain_data[0].shape} | vox_size={self.vox_size}")

        # ---- functional mask(s) ----
        if func_masks is None:
            func_mask_img = compute_multi_brain_mask(func_list, mask_type='whole-brain', threshold=0.5, connected=False)
        else:
            if isinstance(func_masks, (str, Path)):
                func_masks = [func_masks]
            assert len(func_masks) == n_subs, "Number of masks must match number of subjects"
            if n_subs == 1:
                func_mask_img = binarize_img(func_masks[0])
            else:
                func_mask_img = intersect_masks(func_masks, threshold=1.0, connected=False)
        func_mask_bool = self._ensure_bool3d(func_mask_img)
        self.func_mask = func_mask_img  # keep Niimg for resampling

        # ---- ROI mask logic ----
        if roi_masks is None:
            brain_mask_bool = func_mask_bool
            print("[prepare] ROI mask: None → using functional mask")
        elif isinstance(roi_masks, float):
            gm_mask_img = compute_brain_mask(self.func_mask, mask_type='gm', threshold=roi_masks, connected=False)
            gm_bool = self._ensure_bool3d(gm_mask_img)
            assert gm_bool.sum() <= func_mask_bool.sum(), \
                "[prepare] GM mask should have ≤ voxels than functional mask"
            brain_mask_bool = gm_bool & func_mask_bool
            print(f"[prepare] ROI mask: GM probability threshold {roi_masks}")
        else:
            # str or list: ROI mask(s); union them, resample to data grid, intersect with func mask
            if isinstance(roi_masks, list):
                roi_mask_img = intersect_masks(roi_masks, threshold=0.0, connected=False)  # union
                print(f"[prepare] ROI mask: {len(roi_masks)} masks unioned → intersecting with functional mask")
            else:
                roi_mask_img = nib.load(str(roi_masks))
                print(f"[prepare] ROI mask: single ROI mask → intersecting with functional mask")
            roi_resamp = resample_to_img(roi_mask_img, imgs[0], interpolation="nearest")
            roi_bool = self._ensure_bool3d(roi_resamp)
            brain_mask_bool = roi_bool & func_mask_bool

        self.brain_mask = brain_mask_bool

        # sanity checks
        for idx, d in enumerate(self.brain_data):
            assert d.shape[:3] == self.brain_mask.shape, \
                f"[prepare] Subject {idx}: data {d.shape[:3]} != mask {self.brain_mask.shape}"
        assert self.brain_mask.dtype == bool and self.brain_mask.ndim == 3, "[prepare] brain_mask must be 3D boolean"

        # ---- mask coverage precheck ----
        prop_mask, S = self._prop_from_mask(self.brain_mask, shape=self.shape_name, radius=self.radius)
        frac_pass = float((prop_mask >= self.min_prop).mean())
        print(f"[prepare] Shape={self.shape_name} r={self.radius} → neighborhood size S={S}")
        print(f"[prepare] min_prop precheck: fraction of centers passing ≥{self.min_prop:.3f} = {frac_pass:.3f}")
        if frac_pass < 0.01:
            print("[prepare][WARN] With current mask/shape/radius, min_prop is so strict that ~no centers will run.")

        print(f"[prepare] Searchlight configured: shape={self.shape_name}, radius={self.radius}, min_prop={self.min_prop}")

    def run(self, save=True, out_fname='result.nii', save_mask=False):
        """Run the searchlight and optionally save the combined mask."""
        assert self.brain_data is not None and self.brain_mask is not None, \
            "[run] Call prepare() before run()"

        sl = bk_sl.Searchlight(
            sl_rad=self.radius,
            shape=self.shape,
            min_active_voxels_proportion=self.min_prop,  # fraction of cube voxels required to run SL
            max_blk_edge=10,
            pool_size=self.num_sls
        )

        sl.distribute(self.brain_data, self.brain_mask)
        if self.bcvar is not None:
            sl.broadcast(self.bcvar)

        print(f"[run] Running searchlight (pool_size={self.num_sls}) ...")
        sl_result = sl.run_searchlight(self.sl_f)
        print("[run] Searchlight complete.")

        if not save:
            return sl_result

        # Convert to float array & save
        numeric_result = self._object_to_float_array(sl_result)
        save_as_nifti(numeric_result, out_fname, self.affine_matrix, self.vox_size)
        print(f"[run] Saved result to {out_fname}")

        # Optionally save mask
        if save_mask:
            mask_img = nib.Nifti1Image(self.brain_mask.astype(np.uint8), self.affine_matrix)
            out_prefix = out_fname.split('.nii')[0]
            nib.save(mask_img, f"{out_prefix}_mask.nii.gz")
            print(f"[run] Saved mask to {out_prefix}_mask.nii.gz")

        return sl_result
    
def get_searchlight_data(brain_data, sl_mask, verbose=False):
    ''' Returns searchlight data of shape: (num_volumes, num_voxels) '''

    # reshape to 2D: (num_volumes, num_voxels)
    assert brain_data.shape[:3] == sl_mask.shape, f'The brain and mask have different shapes {brain_data.shape} vs {sl_mask.shape}'
    num_vols = brain_data.shape[3]
    assert num_vols in [5, 60, 63], f'Number of volumes is prob. wrong: {num_vols}'
    sl_data = brain_data.reshape(sl_mask.shape[0] * sl_mask.shape[1] * sl_mask.shape[2], num_vols).T

    # remove nans/0 voxels
    # these should be from the 1st level glm noisy voxel exclusions
    try:
        sl_data = VarianceThreshold().fit_transform(sl_data)
    except Exception as e:
        print(f"VarianceThreshold failed: {e}")
        return None

    if verbose:
        print(f"SL data shape: {brain_data.shape}")
        print(f"SL data shape after reshaping: {sl_data.shape}")
        print(f"SL mask shape: {sl_mask.shape}\n")
    return sl_data

#------------------- all trial-pairs location similarity searchlight

def multiple_regression_searchlight(brain_data, sl_mask, myrad, bcvar):

    # extract design matrix
    X = bcvar['X']

    # prepare betas
    betas = get_searchlight_data(brain_data[0], sl_mask)
    if betas is None: # no voxels with variance
        return np.nan
    betas = remove_neutral_trials(betas)

    # prepare neural distances
    y = pdist(betas, metric='correlation').astype(float) # correlation distance

    # fit and return beta
    try:
        model = LinearRegression(fit_intercept=True).fit(X, y)
        return float(model.coef_[0])
    except Exception:
        return np.nan

def run_location_searchlight(
    func_fname,
    mask_fname=None,
    radius=5,
    overwrite=False,
    glm_name="lsa_decision_spm",
):
    """
    Organizer for location searchlight RSA
    Calls multiple_regression_searchlight
    All trial pairs used
    Neural and continuous predictor distances are z-scored across all trial pairs
    """

    # define the predictors
    X_names  = ['location_euc', 'dimension', 'time_linear', 'time_quadratic']

    # organize design matrix
    sub_id   = func_fname.split('/')[-2]
    behavior = load_behavior(sub_id, neutrals=True)
    rdvs     = create_behavioral_rdvs(behavior, remove_neutrals=True, zscore_continuous=True)
    bcvar    = {'X': np.column_stack([np.asarray(rdvs[n], float) for n in X_names]),
                'X_names': X_names}

    # run searchlight kernel
    glm_dir = os.path.join(analysis_dir, glm_name)
    out_dir   = f"{glm_dir}/searchlights/location_time-sq"
    out_fname = f'{out_dir}/{sub_id}_location_time-sq_ball{radius}mm.nii'
    os.makedirs(out_dir, exist_ok=True)
    if not overwrite and os.path.exists(out_fname):
        print(f"File {out_fname} already exists. Skipping computation.")
        return out_fname

    sl = Searchlight(sl_f=multiple_regression_searchlight,
                     shape='ball',
                     radius=radius,
                     min_prop=0.10,
                     num_sls=10)
    sl.prepare(func_fname,
               func_masks=mask_fname,
               roi_masks=None,
               bcvar=bcvar)
    sl.run(save=True, out_fname=out_fname)

#------------------- within-character location similarity searchlight

def within_multiple_regression_searchlight(brain_data, sl_mask, myrad, bcvar):
    """
    Searchlight kernel for WITHIN-character multiple-regression RSA.
    - Uses neural trajectories only (organized by character).
    - Behavioral RDVs and X-matrix are precomputed and passed in via bcvar.
    - Returns OLS beta for 'location_euc'.
    """

    from scipy.spatial.distance import pdist
    from sklearn.linear_model import LinearRegression

    X = bcvar['X']     
    betas         = get_searchlight_data(brain_data[0], sl_mask)
    betas_by_char = organize_by_character(betas)  # list of [K x V]

    y_all = []
    try:
        for char_betas in betas_by_char:
            y = pdist(char_betas, metric='correlation').astype(float)
            y = (y - y.mean()) / (y.std() + 1e-12)  # within-character z
            y_all.append(y)
        y = np.concatenate(y_all, axis=0)
        model = LinearRegression(fit_intercept=True).fit(X, y)
        return float(model.coef_[0])

    except Exception:
        return np.nan

def run_location_within_searchlight(
    func_fname,
    mask_fname=None,
    radius=5,
    overwrite=False,
    glm_name="lsa_decision_spm",
):
    """
    Organizer for WITHIN-character searchlight RSA
    Calls within_multiple_regression_searchlight
    Only within-character trial pairs used
    Neural and continuous predictor distances are z-scored within character
    """

    # define the predictors
    X_names  = ['location_euc', 'dimension', 'time_linear', 'time_quadratic']

    # organize design matrix
    sub_id   = func_fname.split('/')[-2]
    behavior = load_behavior(sub_id, neutrals=True)
    X_cols   = {n: [] for n in X_names}
    for char_beh in organize_by_character(behavior):
        beh_rdvs = create_behavioral_rdvs(char_beh, zscore_continuous=True, remove_neutrals=False)
        for n in X_names:
            X_cols[n].append(np.asarray(beh_rdvs[n], float))
    bcvar = {'X': np.column_stack([np.concatenate(X_cols[n], axis=0) for n in X_names]),
            'X_names': X_names}

    # run searchlight kernel
    glm_dir = os.path.join(analysis_dir, glm_name)
    out_dir   = f"{glm_dir}/searchlights/location-within_time-sq"
    out_fname = f'{out_dir}/{sub_id}_location-within_time-sq_ball{radius}mm.nii'
    os.makedirs(out_dir, exist_ok=True)
    if not overwrite and os.path.exists(out_fname):
        print(f"File {out_fname} already exists. Skipping computation.")
        return out_fname

    sl = Searchlight(sl_f=within_multiple_regression_searchlight,
        shape='ball',
        radius=radius,
        min_prop=0.10,
        num_sls=10
    )
    sl.prepare(func_fname,
                func_masks=mask_fname,
                roi_masks=None,
                bcvar=bcvar)
    sl.run(save=True, out_fname=out_fname)

#------------------- character mean-location searchlight

def character_mean_location_searchlight(brain_data, sl_mask, myrad, bcvar):
    """
    Searchlight kernel for the character-level mean-location model.

    Model:
        neural_character_distance ~ character_mean_location_distance

    Returns the beta on character_mean_location_distance.
    """
    X = bcvar["X"]
    zscore_y = bcvar.get("zscore_y", False)
    character_labels = bcvar["character_labels"]

    # prepare betas
    betas = get_searchlight_data(brain_data[0], sl_mask)
    if betas is None:
        return np.nan
    betas = remove_neutral_trials(betas)

    try:
        # average betas within character -> one neural pattern per character
        unique_chars = np.unique(character_labels)
        char_betas = []
        for ch in unique_chars:
            idx = character_labels == ch
            if idx.sum() < 2:
                continue
            char_betas.append(betas[idx].mean(axis=0))

        if len(char_betas) < 3:
            return np.nan

        char_betas = np.vstack(char_betas)

        # pairwise neural distance across character means
        y = pdist(char_betas, metric="correlation").astype(float)

        if zscore_y:
            y = (y - y.mean()) / (y.std(ddof=0) + 1e-12)

        model = LinearRegression(fit_intercept=True).fit(X, y)
        return float(model.coef_[0])

    except Exception:
        return np.nan

def run_character_mean_location_searchlight(
    func_fname,
    mask_fname=None,
    radius=5,
    overwrite=False,
    zscore_continuous=False,
    zscore_y=False,
    glm_name="lsa_decision_spm",
):
    """
    Organizer for character-level mean-location RSA searchlight.

    Behavioral model:
      1) remove neutral trials
      2) compute one mean affiliation/power coordinate per character
      3) compute between-character Euclidean distances in that mean-location space
      4) regress character-level neural distance on:
            - character mean-location distance

    Parameters
    ----------
    zscore_continuous : bool
        If True, z-score character_mean_location_distance across character pairs.
    zscore_y : bool
        If True, z-score character-level neural distances within each searchlight.
    """
    sub_id = func_fname.split("/")[-2]
    behavior = load_behavior(sub_id, neutrals=True).copy()

    # keep only non-neutral decision trials; this should match remove_neutral_trials(betas)
    keep = behavior["dimension"].isin(["affil", "power"]).to_numpy()
    behavior = behavior.loc[keep].reset_index(drop=True)

    # extract variables
    coords = behavior[["affil_coord", "power_coord"]].to_numpy(float)
    character_labels = behavior["character_role_num"].to_numpy()

    # compute one mean location per character
    unique_chars = np.unique(character_labels)
    char_coords = []
    kept_chars = []

    for ch in unique_chars:
        idx = character_labels == ch
        if idx.sum() < 2:
            continue
        char_coords.append(coords[idx].mean(axis=0))
        kept_chars.append(ch)

    if len(char_coords) < 3:
        raise ValueError(f"{sub_id}: fewer than 3 characters with sufficient trials.")

    char_coords = np.vstack(char_coords)
    kept_chars = np.asarray(kept_chars)

    # behavioral regressor across character pairs
    char_mean_loc_dist = pdist(char_coords, metric="euclidean").astype(float)

    if zscore_continuous:
        char_mean_loc_dist = (
            char_mean_loc_dist - char_mean_loc_dist.mean()
        ) / (char_mean_loc_dist.std(ddof=0) + 1e-12)

    X = char_mean_loc_dist[:, None]
    X_names = ["char_mean_location_dist"]

    bcvar = {
        "X": X,
        "X_names": X_names,
        "character_labels": character_labels,
        "kept_chars": kept_chars,
        "zscore_y": zscore_y,
    }

    # run searchlight kernel
    glm_dir = os.path.join(analysis_dir, glm_name)
    out_dir = f"{glm_dir}/searchlights/char-mean-location"
    out_fname = f"{out_dir}/{sub_id}_char-mean-location_ball{radius}mm.nii"
    os.makedirs(out_dir, exist_ok=True)

    if not overwrite and os.path.exists(out_fname):
        print(f"File {out_fname} already exists. Skipping computation.")
        return out_fname

    sl = Searchlight(
        sl_f=character_mean_location_searchlight,
        shape="ball",
        radius=radius,
        min_prop=0.10,
        num_sls=10,
    )
    sl.prepare(
        func_fname,
        func_masks=mask_fname,
        roi_masks=None,
        bcvar=bcvar,
    )
    sl.run(save=True, out_fname=out_fname)
    return out_fname

#------------------- character mean-location + time-control searchlight

def character_mean_location_timectrl_searchlight(brain_data, sl_mask, myrad, bcvar):
    """
    Searchlight kernel for the character-level mean-location + time-control model.

    Model:
        neural_character_distance ~ character_mean_location_distance + character_mean_onset_distance

    Returns the beta on character_mean_location_distance.
    """
    X = bcvar["X"]
    zscore_y = bcvar.get("zscore_y", False)
    character_labels = bcvar["character_labels"]

    # prepare betas
    betas = get_searchlight_data(brain_data[0], sl_mask)
    if betas is None:
        return np.nan
    betas = remove_neutral_trials(betas)

    try:
        # average betas within character -> one neural pattern per character
        unique_chars = np.unique(character_labels)
        char_betas = []
        for ch in unique_chars:
            idx = character_labels == ch
            if idx.sum() < 2:
                continue
            char_betas.append(betas[idx].mean(axis=0))

        if len(char_betas) < 3:
            return np.nan

        char_betas = np.vstack(char_betas)

        # pairwise neural distance across character means
        y = pdist(char_betas, metric="correlation").astype(float)

        if zscore_y:
            y = (y - y.mean()) / (y.std(ddof=0) + 1e-12)

        model = LinearRegression(fit_intercept=True).fit(X, y)
        return float(model.coef_[0])

    except Exception:
        return np.nan

def run_character_mean_location_timectrl_searchlight(
    func_fname,
    mask_fname=None,
    radius=5,
    overwrite=False,
    zscore_continuous=False,
    zscore_y=False,
    glm_name="lsa_decision_spm",
):
    """
    Organizer for character-level mean-location + time-control RSA searchlight.

    Behavioral model:
      1) remove neutral trials
      2) compute one mean affiliation/power coordinate per character
      3) compute one mean onset per character
      4) regress character-level neural distance on:
            - character mean-location distance
            - character mean-onset distance

    Parameters
    ----------
    zscore_continuous : bool
        If True, z-score continuous regressors across character pairs.
    zscore_y : bool
        If True, z-score character-level neural distances within each searchlight.
    """
    sub_id = func_fname.split("/")[-2]
    behavior = load_behavior(sub_id, neutrals=True).copy()

    # keep only non-neutral decision trials; this should match remove_neutral_trials(betas)
    keep = behavior["dimension"].isin(["affil", "power"]).to_numpy()
    behavior = behavior.loc[keep].reset_index(drop=True)

    # extract variables
    coords = behavior[["affil_coord", "power_coord"]].to_numpy(float)
    character_labels = behavior["character_role_num"].to_numpy()
    onset = behavior["onset"].to_numpy(float)

    # compute one mean location + one mean onset per character
    unique_chars = np.unique(character_labels)
    char_coords = []
    char_onsets = []
    kept_chars = []

    for ch in unique_chars:
        idx = character_labels == ch
        if idx.sum() < 2:
            continue
        char_coords.append(coords[idx].mean(axis=0))
        char_onsets.append(onset[idx].mean())
        kept_chars.append(ch)

    if len(char_coords) < 3:
        raise ValueError(f"{sub_id}: fewer than 3 characters with sufficient trials.")

    char_coords = np.vstack(char_coords)
    char_onsets = np.asarray(char_onsets, float)
    kept_chars = np.asarray(kept_chars)

    # behavioral regressors across character pairs
    char_mean_loc_dist = pdist(char_coords, metric="euclidean").astype(float)
    char_mean_onset_dist = pdist(char_onsets[:, None], metric="euclidean").astype(float)

    if zscore_continuous:
        char_mean_loc_dist = (
            char_mean_loc_dist - char_mean_loc_dist.mean()
        ) / (char_mean_loc_dist.std(ddof=0) + 1e-12)

        char_mean_onset_dist = (
            char_mean_onset_dist - char_mean_onset_dist.mean()
        ) / (char_mean_onset_dist.std(ddof=0) + 1e-12)

    X = np.column_stack([char_mean_loc_dist, char_mean_onset_dist])
    X_names = ["char_mean_location_dist", "char_mean_onset_dist"]

    bcvar = {
        "X": X,
        "X_names": X_names,
        "character_labels": character_labels,
        "kept_chars": kept_chars,
        "zscore_y": zscore_y,
    }

    # run searchlight kernel
    glm_dir = os.path.join(analysis_dir, glm_name)
    out_dir = f"{glm_dir}/searchlights/char-mean-location-timectrl"
    out_fname = f"{out_dir}/{sub_id}_char-mean-location-timectrl_ball{radius}mm.nii"
    os.makedirs(out_dir, exist_ok=True)

    if not overwrite and os.path.exists(out_fname):
        print(f"File {out_fname} already exists. Skipping computation.")
        return out_fname

    sl = Searchlight(
        sl_f=character_mean_location_timectrl_searchlight,
        shape="ball",
        radius=radius,
        min_prop=0.10,
        num_sls=10,
    )
    sl.prepare(
        func_fname,
        func_masks=mask_fname,
        roi_masks=None,
        bcvar=bcvar,
    )
    sl.run(save=True, out_fname=out_fname)
    return out_fname

#------------------- dimension similarity searchlight

def pattern_similarity_searchlight(brain_data, sl_mask, myrad, bcvar):
    '''
        Run an average pattern similarity analysis in a brainiak searchlight 
        Returns the difference in pattern similarity between the two conditions
    '''

    # get predicted rdv
    pred_rsv = bcvar 

    # get searchlight volume
    betas = get_searchlight_data(brain_data[0], sl_mask)
    betas = remove_neutral_trials(betas)

    # use Fisher's z-transformed Pearsons correlation for pattern similarity
    neural_rsv = 1 - pdist(betas, metric='correlation')
    neural_rsv = np.arctanh(np.clip(neural_rsv, -0.999999, 0.999999)) # Fisher's z transform... this doesn't make a difference

    # calculate pattern similarity difference
    return np.mean(neural_rsv[pred_rsv==1]) - np.mean(neural_rsv[pred_rsv==0]) 

def run_dimension_searchlight(func_fname, 
                              mask_fname, 
                              radius=5, 
                              overwrite=False,
                              glm_name="lsa_onset"):
    
    sub_id  = func_fname.split('/')[-2]

    # dimension pattern similarity: mean(ps within dimension) - mean(ps between dimensions)
    affil     = remove_neutral_trials((decision_trials['dimension'] == 'affil').values.astype(int))[:, np.newaxis]
    dimn_rsv = 1 - pdist(affil, metric='hamming')

    # run searchlight
    onset_glm_dir = os.path.join(analysis_dir, glm_name)
    out_dir = f"{onset_glm_dir}/searchlights/dimension"
    os.makedirs(out_dir, exist_ok=True)
    out_fname = f'{out_dir}/{sub_id}_dimension_ball{radius}mm.nii'
    if not overwrite and os.path.exists(out_fname):
        print(f"File {out_fname} already exists. Skipping computation.")
        return out_fname
    
    sl = Searchlight(sl_f=pattern_similarity_searchlight, 
                     shape='ball', 
                     radius=radius, 
                     min_prop=0.10, 
                     num_sls=10)
    sl.prepare(func_fname, func_masks=mask_fname, bcvar=dimn_rsv)    
    sl.run(save=True, out_fname=out_fname)

#-------------------dimension regression searchlight

def dimension_regression_searchlight(brain_data, sl_mask, myrad, bcvar):
    """
    Multiple-regression searchlight RSA for dimension.

    Returns the adjusted dimension beta:
        neural correlation distance ~ dimension_mismatch
                                   + location_euc
                                   + time_linear
                                   + time_quadratic

    Positive beta means:
        between-dimension pairs have larger neural distance than within-dimension pairs.
    """

    # -----------------------------
    # extract design matrix
    # -----------------------------
    X = bcvar["X"]

    # -----------------------------
    # prepare searchlight betas
    # -----------------------------
    betas = get_searchlight_data(brain_data[0], sl_mask)

    if betas is None:
        return np.nan

    betas = remove_neutral_trials(betas)

    # need at least a few voxels with variance
    if betas.ndim != 2 or betas.shape[1] < 2:
        return np.nan

    # remove bad voxels
    good_vox = np.isfinite(betas).all(axis=0) & (np.nanstd(betas, axis=0) > 0)
    betas = betas[:, good_vox]

    if betas.shape[1] < 2:
        return np.nan

    # -----------------------------
    # neural RDV: correlation distance
    # -----------------------------
    y = pdist(betas, metric="correlation").astype(float)

    # -----------------------------
    # fit regression
    # -----------------------------
    good = np.isfinite(y) & np.isfinite(X).all(axis=1)

    if good.sum() < X.shape[1] + 2:
        return np.nan

    try:
        model = LinearRegression(fit_intercept=True).fit(X[good], y[good])

        # coef_[0] is dimension because dimension is first in X_names
        return float(model.coef_[0])

    except Exception:
        return np.nan

def run_dimension_regression_searchlight(
    func_fname,
    mask_fname=None,
    radius=5,
    overwrite=False,
    glm_name="lsa_decision_spm",
):
    """
    Organizer for dimension regression searchlight RSA.

    This matches the location regression analysis:
      - neural target is correlation distance
      - all trial pairs are used
      - continuous behavioral RDVs are z-scored across trial pairs
      - controls are location_euc, time_linear, and time_quadratic

    Output map:
      beta for dimension_mismatch, adjusted for location and time.
    """

    # -----------------------------
    # subject and behavior
    # -----------------------------
    sub_id = func_fname.split("/")[-2]

    behavior = load_behavior(sub_id, neutrals=True)

    # create standard behavioral RDVs
    rdvs = create_behavioral_rdvs(
        behavior,
        remove_neutrals=True,
        zscore_continuous=True,
    )

    # -----------------------------
    # define dimension predictor explicitly
    # -----------------------------
    # 0 = same dimension
    # 1 = different dimension
    #
    # This makes the sign easy to interpret:
    # positive beta = larger neural distance between dimensions.
    dim = remove_neutral_trials(
        (decision_trials["dimension"] == "affil").values.astype(int)
    )[:, np.newaxis]

    dimension_mismatch = pdist(dim, metric="hamming").astype(float)

    # -----------------------------
    # regression design
    # -----------------------------

    X_names = [
        "dimension_mismatch",
        "location_euc",
        "time_linear",
        "time_quadratic",
    ]

    X = np.column_stack([
        dimension_mismatch,
        np.asarray(rdvs["location_euc"], float),
        np.asarray(rdvs["time_linear"], float),
        np.asarray(rdvs["time_quadratic"], float),
    ])

    bcvar = {
        "X": X,
        "X_names": X_names,
    }

    # -----------------------------
    # output path
    # -----------------------------

    glm_dir = os.path.join(analysis_dir, glm_name)

    out_dir = f"{glm_dir}/searchlights/dimension-regression_location_time-sq"
    os.makedirs(out_dir, exist_ok=True)

    out_fname = f"{out_dir}/{sub_id}_dimension-regression_location_time-sq_ball{radius}mm.nii"

    if not overwrite and os.path.exists(out_fname):
        print(f"File {out_fname} already exists. Skipping computation.")
        return out_fname

    # -----------------------------
    # run searchlight
    # -----------------------------
    
    sl = Searchlight(
        sl_f=dimension_regression_searchlight,
        shape="ball",
        radius=radius,
        min_prop=0.10,
        num_sls=10,
    )

    sl.prepare(
        func_fname,
        func_masks=mask_fname,
        roi_masks=None,
        bcvar=bcvar,
    )

    sl.run(save=True, out_fname=out_fname)

    return out_fname

#------------------- choice-related searchlights

def get_choice_vectors(sub_id, remove_neutrals=True):
    """
    Return trialwise signed 2D choice vectors:
        A+ = [ 1,  0]
        A- = [-1,  0]
        P+ = [ 0,  1]
        P- = [ 0, -1]

    Uses subject-specific behavior, so exact choice and sign reflect
    the participant's actual choices.
    """
    behavior = load_behavior(sub_id, neutrals=True).copy()
    choice_vec = behavior[['affil_decision', 'power_decision']].to_numpy(float)

    if remove_neutrals:
        if choice_vec.shape[0] == 63:
            choice_vec = remove_neutral_trials(choice_vec)
        else:
            keep = behavior['dimension'].isin(['affil', 'power']).to_numpy()
            choice_vec = choice_vec[keep]

    return choice_vec

def choice_vector_distance_searchlight(brain_data, sl_mask, myrad, bcvar):
    """
    Searchlight kernel for 2D signed choice-vector distance.

    Model:
        neural_distance ~ choice_vector_distance

    Returns beta on choice_vector_distance.
    Positive beta means neural patterns are more dissimilar when
    signed 2D choice vectors are farther apart.
    """

    X = bcvar['X']

    betas = get_searchlight_data(brain_data[0], sl_mask)
    if betas is None:
        return np.nan
    betas = remove_neutral_trials(betas)

    y = pdist(betas, metric='correlation').astype(float)

    try:
        model = LinearRegression(fit_intercept=True).fit(X, y)
        return float(model.coef_[0])
    except Exception:
        return np.nan

def _build_choice_control_bcvar(sub_id, main_regressor, main_name, include_dimension=True):
    """
    Build a minimal design matrix for choice-related RSA:
        neural_distance ~ main_regressor + time_linear (+ dimension)

    main_regressor should already be a condensed-form trial-pair vector.
    """
    behavior = load_behavior(sub_id, neutrals=True)
    rdvs = create_behavioral_rdvs(
        behavior,
        zscore_continuous=True,
        remove_neutrals=True,
    )

    X_names = [main_name, 'time_linear']
    X_cols = [np.asarray(main_regressor, float), np.asarray(rdvs['time_linear'], float)]

    if include_dimension:
        X_names.append('dimension')
        X_cols.append(np.asarray(rdvs['dimension'], float))

    return {
        'X': np.column_stack(X_cols),
        'X_names': X_names,
    }

def run_exact_choice_searchlight(func_fname,
                                 mask_fname=None,
                                 radius=5,
                                 overwrite=False,
                                 glm_name="lsa_onset"):
    """
    Exact choice RSA with nuisance controls.

    Model:
        neural_distance ~ exact_choice_diff + time_linear + dimension

    exact_choice_diff:
        0 = same exact choice
        1 = different exact choice
    """
    sub_id = func_fname.split('/')[-2]

    choice_vec = get_choice_vectors(sub_id, remove_neutrals=True)
    choice_code = (2 * choice_vec[:, 0] + choice_vec[:, 1]).astype(int)
    exact_choice_diff = pdist(choice_code[:, np.newaxis], metric='hamming').astype(float)

    bcvar = _build_choice_control_bcvar(
        sub_id,
        main_regressor=exact_choice_diff,
        main_name='exact_choice_diff',
        include_dimension=True,
    )

    glm_dir = os.path.join(analysis_dir, glm_name)
    out_dir = f"{glm_dir}/searchlights/exact-choice"
    os.makedirs(out_dir, exist_ok=True)
    out_fname = f'{out_dir}/{sub_id}_exact-choice_ball{radius}mm.nii'

    if not overwrite and os.path.exists(out_fname):
        print(f"File {out_fname} already exists. Skipping computation.")
        return out_fname

    sl = Searchlight(sl_f=multiple_regression_searchlight,
                     shape='ball',
                     radius=radius,
                     min_prop=0.10,
                     num_sls=10)
    sl.prepare(func_fname, func_masks=mask_fname, bcvar=bcvar)
    sl.run(save=True, out_fname=out_fname)
    return out_fname

def run_same_sign_searchlight(func_fname,
                              mask_fname=None,
                              radius=5,
                              overwrite=False,
                              glm_name="lsa_onset"):
    """
    Same-sign RSA with nuisance controls.

    Model:
        neural_distance ~ sign_diff + time_linear + dimension

    sign_diff:
        0 = same sign
        1 = different sign
    """
    sub_id = func_fname.split('/')[-2]

    choice_vec = get_choice_vectors(sub_id, remove_neutrals=True)
    choice_sign = np.sign(choice_vec.sum(axis=1)).astype(int)
    sign_diff = pdist(choice_sign[:, np.newaxis], metric='hamming').astype(float)

    bcvar = _build_choice_control_bcvar(
        sub_id,
        main_regressor=sign_diff,
        main_name='sign_diff',
        include_dimension=True,
    )

    glm_dir = os.path.join(analysis_dir, glm_name)
    out_dir = f"{glm_dir}/searchlights/same-sign"
    os.makedirs(out_dir, exist_ok=True)
    out_fname = f'{out_dir}/{sub_id}_same-sign_ball{radius}mm.nii'

    if not overwrite and os.path.exists(out_fname):
        print(f"File {out_fname} already exists. Skipping computation.")
        return out_fname

    sl = Searchlight(sl_f=multiple_regression_searchlight,
                     shape='ball',
                     radius=radius,
                     min_prop=0.10,
                     num_sls=10)
    sl.prepare(func_fname, func_masks=mask_fname, bcvar=bcvar)
    sl.run(save=True, out_fname=out_fname)
    return out_fname

def run_choice_vector_distance_searchlight(func_fname,
                                           mask_fname=None,
                                           radius=5,
                                           overwrite=False,
                                           zscore_continuous=True,
                                           glm_name="lsa_onset"):
    """
    2D choice-vector-distance RSA with nuisance controls.

    Model:
        neural_distance ~ choice_vector_distance + time_linear + dimension
    """
    sub_id = func_fname.split('/')[-2]

    choice_vec = get_choice_vectors(sub_id, remove_neutrals=True)
    choice_dist = pdist(choice_vec, metric='euclidean').astype(float)

    if zscore_continuous:
        choice_dist = (choice_dist - choice_dist.mean()) / (choice_dist.std() + 1e-12)

    bcvar = _build_choice_control_bcvar(
        sub_id,
        main_regressor=choice_dist,
        main_name='choice_vector_distance',
        include_dimension=True,
    )

    glm_dir = os.path.join(analysis_dir, glm_name)
    out_dir = f"{glm_dir}/searchlights/choice-vector-distance"
    os.makedirs(out_dir, exist_ok=True)
    out_fname = f'{out_dir}/{sub_id}_choice-vector-distance_ball{radius}mm.nii'

    if not overwrite and os.path.exists(out_fname):
        print(f"File {out_fname} already exists. Skipping computation.")
        return out_fname

    sl = Searchlight(sl_f=choice_vector_distance_searchlight,
                     shape='ball',
                     radius=radius,
                     min_prop=0.10,
                     num_sls=10)
    sl.prepare(func_fname, func_masks=mask_fname, bcvar=bcvar)
    sl.run(save=True, out_fname=out_fname)
    return out_fname

########################################################################
# SECOND LEVEL
########################################################################


def find_sl_imgs(sl_dir, fname_pattern):

    sl_niis = glob.glob(f'{sl_dir}/*{fname_pattern}*')
    sl_niis = [
        s for s in sl_niis 
        if s.split('/')[-1].split('_')[0].removeprefix("sub-") in incl_subs
    ]
    print(f'Found {len(sl_niis)} searchlight images')

    # split sl_niis into two samples if sub_id > 2 or not
    sl_dict = {'Initial': {'sub_ids':[], 'imgs':[]}, 
               'Validation': {'sub_ids':[], 'imgs':[]}, 
               'Combined': {'sub_ids':[], 'imgs':[]}}
    for sl_nii in sl_niis:
        sub_id = sl_nii.split('/')[-1].split('_')[0].removeprefix("sub-")
        sample = 'Validation' if len(sub_id) > 2 else 'Initial'
        sl_dict[sample]['sub_ids'].append(sub_id)        
        sl_dict[sample]['imgs'].append(sl_nii)
        sl_dict['Combined']['sub_ids'].append(sub_id) 
        sl_dict['Combined']['imgs'].append(sl_nii)
        
    print(f'Initial n = {len(sl_dict["Initial"]["sub_ids"])}')
    print(f'Validation n = {len(sl_dict["Validation"]["sub_ids"])}')
    return sl_dict

def rt_diff_for(sub_id):
    behav = load_behavior(sub_id, neutrals=False)
    behav = behav.loc[behav['responded']]
    m = behav.groupby('dimension')['reaction_time'].mean()
    return abs((m.get('affil', np.nan) - m.get('power', np.nan)))

def fd_mean_for(sub_id):
    sel = data.loc[data['sub_id'] == sub_id, 'mean_fd']
    return float(sel.iloc[0]) if not sel.empty else np.nan

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

def run_tfce_correction(sl_model, fwhm=8, overwrite=False):

    sl_dir    = f'../analyses/lsa_rt/searchlights/{sl_model}/'
    sl_dict   = find_sl_imgs(sl_dir, f'{sl_model}_ball5mm')
    sub_ids   = sl_dict['Combined']['sub_ids']
    imgs      = sl_dict['Combined']['imgs']
    mask_img  = f'{mask_dir}/GM.nii.gz'
    # mask_img  = f'{mask_dir}/L-HPC_harvardoxford_maxprob-thr25_1mm.nii'


    # load if it exists
    fname = f'{sl_dir}/../{sl_model}_ball5mm_{fwhm}fwhm_n{len(sub_ids)}_TFCE.nii.gz'
    if os.path.exists(fname) and not overwrite:
        print('Already ran...')

    # run it if not
    else:

        # control for things
        confound_df = pd.DataFrame([
            {
                'subject_label': img,
                'sub_id': sub_id,
                'sample': int(len(str(sub_id)) > 2), # 0 = Initial, 1 = Validation
                'rt': rt_diff_for(sub_id),           # |mean RT_affil − mean RT_power|
                'fd_mean': fd_mean_for(sub_id),      # motion covariate
            }
            for sub_id, img in zip(sub_ids, imgs)
        ])

        # do within-sample scaling
        for g in (0, 1):
            idx = confound_df['sample'].astype(int).eq(g)

            rt_mu  = np.mean(confound_df.loc[idx, 'rt'])
            rt_sd  = np.std(confound_df.loc[idx, 'rt'], ddof=1)
            confound_df.loc[idx, 'rt'] = (confound_df.loc[idx, 'rt'] - rt_mu) / rt_sd

            fd_mu = np.mean(confound_df.loc[idx, 'fd_mean'])
            fd_sd = np.std(confound_df.loc[idx, 'fd_mean'], ddof=1) 
            confound_df.loc[idx, 'fd_mean'] = (confound_df.loc[idx, 'fd_mean'] - fd_mu) / fd_sd

        # build design matrix
        sample_dummies = pd.get_dummies(confound_df['sample'], prefix='sample', drop_first=True).astype(float)
        dm_input       = pd.concat([confound_df[['subject_label', 'rt']], sample_dummies], axis=1)
        design_matrix  = make_second_level_design_matrix(subjects_label=imgs, confounds=dm_input)

        # make intercept into the n-weighted grand mean so we can interpret the intercept as the grand mean across samples
        if 'sample_1' in design_matrix.columns:
            design_matrix['sample_centered'] = design_matrix['sample_1'] - design_matrix['sample_1'].mean()
            design_matrix.drop(columns=['sample_1'], inplace=True)

        # run the permutation t-test & return voxelwise corrected -log10(p) image
        ttest_dict = compute_permutation_ttest(imgs, mask_img=mask_img,
                                            design_matrix=design_matrix, second_level_contrast='intercept', 
                                            n_perm=5000, two_sided=False, model_intercept=False,
                                            threshold=None, tfce=True, fwhm=fwhm)
        ttest_img = ttest_dict['logp_max_tfce']
        ttest_img.to_filename(fname)


########################################################################
# OTHER SEARCHLIGHTS....
########################################################################


#------------------- between character analysis

def pairwise_trajectory_distances(trajs, *, 
                                  feature_metric='euclidean',
                                  align='elementwise', 
                                  reducer=np.nanmean):
    """
    Compute an [C x C] symmetric matrix of trajectory distances across C trajectories.

    trajs: array-like of shape [C, K, D] or list of KxD arrays.
    """
    
    def trajectory_pair_distance(A, B, *, 
                                feature_metric='euclidean',
                                align='elementwise', 
                                reducer=np.nanmean):
        """
        Distance between two trajectories A,B: [K x D].

        feature_metric: 'euclidean' | 'correlation' | 'cosine'
        align:          'elementwise' (index-aligned) | 'allpairs' (pairwise)
        reducer:        function to reduce a vector/matrix to a scalar (default: nanmean)

        Returns a scalar distance.
        """
        
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        if align == 'elementwise':
            if A.shape[0] != B.shape[0]:
                raise ValueError("Elementwise alignment requires equal timepoints")
            D = cdist(A, B, metric=feature_metric)
            return float(reducer(np.diag(D)))

        elif align == 'allpairs':
            # average over all k×k' distances
            D = cdist(A, B, metric=feature_metric)
            return float(reducer(D))
    
    trajs = np.asarray(trajs, dtype=object if isinstance(trajs, list) else float)
    C = len(trajs)
    out = np.zeros((C, C), dtype=float)

    for i, j in itertools.combinations(range(C), 2):
        d = trajectory_pair_distance(trajs[i], trajs[j],
                                     feature_metric=feature_metric,
                                     align=align, 
                                     reducer=reducer)
        out[i, j] = out[j, i] = d
    return out

def fit_subject_trajs(roi_betas, 
                      behav_data, *,
                      neural_metric='correlation', 
                      behav_metric='euclidean',
                      align='allpairs', 
                      reducer=np.mean):

    # -------- organize trajectories by character --------

    neural_trajs = organize_by_character(roi_betas)  # list of [5 x V]
    behav_trajs  = organize_by_character(behav_data[['affil_coord','power_coord']].to_numpy())  # list of [5 x 2]

    # -------- RDVs  --------

    beh_M    = pairwise_trajectory_distances(behav_trajs,
                                             feature_metric=behav_metric,
                                             align=align, 
                                             reducer=reducer)
    neu_M    = pairwise_trajectory_distances(neural_trajs,
                                             feature_metric=neural_metric,
                                             align=align, 
                                             reducer=reducer)
    beh_rdv  = flatten_upper_tri(beh_M)
    neur_rdv = flatten_upper_tri(neu_M)

    # -------- Nuisance RDVs --------

    # Mean |Δtime| per character pair (time-only control)
    chars  = [1, 2, 3, 4, 5]
    C      = len(chars)
    time_M  = np.zeros((C, C))

    # Per pair, compute aligned means over k (EW) or full (PW) means.
    for i, j in itertools.combinations(range(C), 2):
        mi = (behav_data['character_role_num'] == chars[i]).values
        mj = (behav_data['character_role_num'] == chars[j]).values
        Xi = behav_data.loc[mi, ['onset']].reset_index(drop=True)
        Xj = behav_data.loc[mj, ['onset']].reset_index(drop=True)
        if align == 'elementwise':
            dtime = np.abs(Xi['onset'].to_numpy() - Xj['onset'].to_numpy())
        else:
            dtime = np.abs(Xi['onset'].to_numpy()[:,None] - Xj['onset'].to_numpy()[None,:])
        time_M[i,j]  = time_M[j,i]  = np.nanmean(dtime)
    time_rdv  = flatten_upper_tri(time_M)

    # partial Kendall via residualization
    def _resid(y, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.column_stack([np.ones(len(y)), X])
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

    conf_rdvs = np.column_stack([time_rdv])
    neur_rdv  = _resid(neur_rdv, conf_rdvs)
    beh_rdv   = _resid(beh_rdv,  conf_rdvs)
    
    return scipy.stats.kendalltau(beh_rdv, neur_rdv, nan_policy='omit')[0]

def trajs_searchlight(brain_data, sl_mask, myrad, bcvar):
    """
    Searchlight kernel for same-kind trajectory RDV RSA.
    - Calls fit_subject_trajs on sphere betas + preloaded behav_data.
    - Returns Kendall's tau between behavioral and neural character-pair RDVs.
    """
    try:
        betas = get_searchlight_data(brain_data[0], sl_mask)
        tau = fit_subject_trajs(betas, 
                                bcvar['behav_data'],
                                neural_metric='correlation',
                                behav_metric='euclidean',
                                align='allpairs')
        return tau 

    except Exception:
        return np.nan

def run_trajs_searchlight(func_fname, mask_fname=None, radius=5):
    """
    Runner for same-kind trajectory RDV RSA using fit_subject_trajs.
    - Loads behavior once and passes it into the kernel via bcvar.
    - Uses defaults matching fit_subject_trajs (correlation/euclidean, allpairs, controls on).
    """

    # subject & behavior
    sub_id   = func_fname.split('/')[-2]
    behavior = load_behavior(sub_id, neutrals=True)
    bcvar = {'behav_data': behavior}

    # output dir & name
    out_dir = f"/sc/arion/projects/OlfMem/mgs/analyses/lsa_rt/searchlights/trajs"
    os.makedirs(out_dir, exist_ok=True)

    # run searchlight
    sl = Searchlight(
        sl_f=trajs_searchlight,
        shape='ball',
        radius=radius,
        min_prop=0.10,
        num_sls=10
    )
    sl.prepare(
        func_fname,
        func_masks=mask_fname,
        roi_masks=None,
        bcvar=bcvar
    )
    sl.run(save=True,
           out_prefix=f"{out_dir}/{sub_id}_trajs_ball{radius}mm")

#------------------- leave one character out searchlights

def loco_searchlight(brain_data, sl_mask, myrad, bcvar):
    """
    bcvar:
      - 'X'        : [N_pairs x P] matrix of RDVs in condensed pdist order
                     (col 0 MUST be 'location_euc'; remaining columns are confounds)
      - 'X_names'  : list of column names for X (X_names[0] == 'location_euc')
      - 'char_ids' : length-T vector of character id per trial (after removing neutrals)

    Logic (per fold):
      1) Define test pairs (HH) where both endpoints are the held character; train pairs (OO) where neither endpoint is held.
      2) Rank-transform (Spearman; method='average') y (neural), x (location_euc), and
         continuous confounds **within train/test splits**; do NOT rank categorical confounds.
      3) Residualize y and x on confounds using train-only fits; apply to test.
      4) Fit y_res ~ x_res on train; predict in test; evaluate ρ = Spearman(y_te_res, ŷ_te_res).
      5) Fisher‑z transform ρ, average across folds, tanh back.
    """
    
    # --- RDVs from bcvar ---
    X        = np.asarray(bcvar['X'], float)
    X_names  = list(bcvar['X_names'])
    char_ids = np.asarray(bcvar['char_ids'])

    # --- neural distances for this sphere ---
    betas = get_searchlight_data(brain_data[0], sl_mask)
    betas = remove_neutral_trials(betas)
    y_all = pdist(betas, metric='correlation').astype(float)

    # target regressor and confounds
    x_all = X[:, 0].astype(float) # 'location_euc'
    C_all = X[:, 1:].astype(float) if X.shape[1] > 1 else None
    C_names = X_names[1:]

    # which confounds are continuous (rank) vs categorical (no rank)
    do_not_rank = {'dimension', 'character'}    # extend if needed
    is_cont     = [(n not in do_not_rank) for n in C_names]

    # pair indexing
    T = len(char_ids)
    i_idx, j_idx = np.triu_indices(T, 1)

    # aggregate across folds
    z_rhos = []
    for held in np.unique(char_ids):

        # training trials are other characters only, test trials are held character only
        # drop the mixed held-other pairs so no leakage
        held_idx   = np.where(char_ids == held)[0]
        in_i       = np.isin(i_idx, held_idx)
        in_j       = np.isin(j_idx, held_idx)
        test_mask  = (in_i & in_j)        # HH (held out-held out pairs)
        train_mask = ~(in_i | in_j)       # OO (other characters only pairs)
        y_tr, y_te = y_all[train_mask], y_all[test_mask]
        x_tr, x_te = x_all[train_mask], x_all[test_mask]
        if C_all is not None and C_all.size:
            C_tr_raw, C_te_raw = C_all[train_mask], C_all[test_mask]
        else:
            C_tr_raw = C_te_raw = None

        # ----- Rank-transform (within split) -----
        # only rank continuous variables; categorical are already 0/1
        y_tr = rankdata(y_tr, method='average').astype(float)
        y_te = rankdata(y_te, method='average').astype(float)
        x_tr = rankdata(x_tr, method='average').astype(float)
        x_te = rankdata(x_te, method='average').astype(float)
        C_tr_cols, C_te_cols = [], []
        for j, cont in enumerate(is_cont):
            if cont:
                C_tr_cols.append(rankdata(C_tr_raw[:, j], method='average').astype(float))
                C_te_cols.append(rankdata(C_te_raw[:, j], method='average').astype(float))
            else:
                C_tr_cols.append(C_tr_raw[:, j].astype(float))
                C_te_cols.append(C_te_raw[:, j].astype(float))
        C_tr = np.column_stack(C_tr_cols)
        C_te = np.column_stack(C_te_cols)

        # ----- Remove confounds: residualize y and x on confounds (fit on train, apply to test) -----
        reg_y = LinearRegression().fit(C_tr, y_tr)
        reg_x = LinearRegression().fit(C_tr, x_tr)
        y_tr_res = y_tr - reg_y.predict(C_tr)   
        y_te_res = y_te - reg_y.predict(C_te)
        x_tr_res = x_tr - reg_x.predict(C_tr)   
        x_te_res = x_te - reg_x.predict(C_te)

        # ----- Learn mapping on OO & predict in HH -----
        # uses the residualized x and y to ensure no confound influence
        map_reg     = LinearRegression().fit(x_tr_res.reshape(-1,1), y_tr_res)
        yhat_te_res = map_reg.predict(x_te_res.reshape(-1,1))

        # ----- Correlate predicted and actual in test -----
        # Spearman ρ (rank correlation) b/c we care about relative distances
        # Fisher-z transform ρ before averaging across folds
        # if this is significant across folds, suggests the mapping learned in OO generalizes to HH
        rho = scipy.stats.spearmanr(y_te_res, yhat_te_res).correlation
        rho = np.clip(rho, -0.999999, 0.999999)
        z_rhos.append(np.arctanh(rho)); 

    # mean Fisher-z ρ across folds, tanh back to correlation
    return np.tanh(np.mean(z_rhos))

def run_location_loco_searchlight(func_fname, mask_fname=None, radius=5):
    """
    Prepares bcvar (X, X_names, char_ids) and runs a LOCO searchlight that
    returns a map of Fisher-z mean Spearman ρ across characters.
    """

    # --- compute behavioral RDVs once (no pre-ranking; handled in worker) ---
    sub_id   = func_fname.split('/')[-2]
    behavior = load_behavior(sub_id, neutrals=True)
    rdvs     = create_behavioral_rdvs(behavior, rank_continuous=False, zscore_continuous=False)

    # --- organize design matrix ---
    control_names = ['dimension', 'familiarity_linear', 'time_linear', 'time_quadratic']
    X_names = ['location_euc'] + [n for n in control_names if n in rdvs]
    X       = np.column_stack([np.asarray(rdvs[n], float) for n in X_names])

    # --- LOCO split ids (trial-level) ---
    char_ids = remove_neutral_trials(behavior['character_role_num'].values)
    bcvar    = {'X': X, 'X_names': X_names, 'char_ids': char_ids}

    # --- setup output dir & prefix  ---
    parts   = func_fname.split(os.sep)
    base    = os.sep.join(parts[:parts.index("mgs")+1])
    out_dir = f"{base}/analyses/searchlights/location_loco"
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = f"{out_dir}/{sub_id}_location_loco_ball{radius}mm"

    # --- run searchlight ---
    sl = Searchlight(
        sl_f=loco_searchlight,
        shape='ball',
        radius=radius,
        min_prop=0.10,
        num_sls=10
    )
    sl.prepare(func_fname, func_masks=mask_fname, roi_masks=None, bcvar=bcvar)
    sl.run(save=True, out_prefix=out_prefix)

#------------------- semantic similarity searchlights

def run_semantic_searchlight(func_fname, mask_fname, embd_fname, radius=5):

    # compute behavioral RDVs once
    sub_id   = func_fname.split('/')[-2]
    behavior = load_behavior(sub_id, neutrals=True)
    rdvs     = create_behavioral_rdvs(behavior, rank_continuous=True, zscore_continuous=True)

    # add in the semantic embeddings
    embds   = remove_neutral_trials(np.load(embd_fname)['embeddings'])
    sem_rdv = pdist(embds, metric='cosine').astype(float)                  # cosine distance
    sem_rdv = rankdata(sem_rdv, method='average').astype(float)            # rank
    sem_rdv = (sem_rdv - sem_rdv.mean()) / (sem_rdv.std() + 1e-12)         # z-score
    rdvs['choice_semantics'] = sem_rdv

    # organize design matrix
    control_names = [
        'location_euc',
        'dimension', 'character', 'scene',
        'familiarity_linear', 
        'time_linear', 'time_quadratic'
    ]
    X_names = ['choice_semantics'] + control_names # the regressor of interest is first
    X = np.column_stack([np.asarray(rdvs[n], float) for n in X_names])
    bcvar = {'X': X, 'X_names': X_names}

    # setup output dir & fname
    embd_label = os.path.basename(embd_fname).rsplit('_', 2)[-2]
    parts   = func_fname.split(os.sep)
    base    = os.sep.join(parts[:parts.index("mgs")+1])
    out_dir = f"{base}/analyses/searchlights/{embd_label}"
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = f"{out_dir}/{sub_id}_{embd_label}_ball{radius}mm"

    # run searchlight kernel
    sl = Searchlight(sl_f=multiple_regression_searchlight, 
                     shape='ball', 
                     radius=radius,
                     min_prop=0.10, 
                     num_sls=10)
    sl.prepare(func_fname, 
               func_masks=mask_fname, 
               roi_masks=None, 
               bcvar=bcvar)
    sl.run(save=True, out_prefix=out_prefix)

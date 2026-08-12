import zlib
import sys, warnings, os, glob, copy, itertools, json, shutil, math, re
from IPython.display import Markdown
from tqdm import tqdm
import patsy
from datetime import date
from pathlib import Path
from collections import Counter
from re import search
from six.moves import cPickle as pickle
from nilearn import datasets, image
from nilearn import image as nimg
if not sys.warnoptions: warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
from numpy.linalg import norm

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Patch
from mpl_toolkits import axes_grid1
import matplotlib.backends.backend_pdf
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import statsmodels.api as sm
from statsmodels.multivariate.cancorr import CanCorr

import scipy
from scipy import optimize
from scipy.stats import zscore, chi2_contingency, wilcoxon, kendalltau, pearsonr, spearmanr, kendalltau, ttest_1samp, wilcoxon
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, TransformerMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, KNeighborsRegressor, NeighborhoodComponentsAnalysis, RadiusNeighborsClassifier, RadiusNeighborsRegressor
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold, StratifiedKFold, train_test_split, StratifiedShuffleSplit, LeaveOneOut
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, pairwise_distances, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, adjusted_mutual_info_score
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression

import nibabel as nib

import nilearn
from nilearn import image, plotting
from nilearn.glm import cluster_level_inference, threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiSpheresMasker, NiftiMapsMasker
from nilearn.image import load_img, get_data, new_img_like, math_img
from nilearn.masking import compute_brain_mask
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

import scipy.stats as st
from typing import Sequence, Literal, Tuple, Optional
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist
from scipy.stats import kendalltau
from scipy.spatial.distance import pdist, squareform


from pathlib import Path
import os

def _find_project_root(start: Path) -> Path:
    """
    Walk upward from this file to find the repo/project root.
    Heuristic: first parent containing expected top-level folders.
    Optional override via env var: SNT_PROJECT_ROOT.
    """
    env_root = os.environ.get("SNT_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    # Heuristic markers (adjust if your repo uses different names)
    must_exist = ("data", "results")     # strongest signals
    any_exist  = ("figures", "masks", "analyses", ".git")

    for p in [start.parent, *start.parents]:
        if all((p / m).exists() for m in must_exist) and any((p / m).exists() for m in any_exist):
            return p.resolve()

    # Fallback: assume this file lives one level down from root (old '..' behavior)
    return start.parent.parent.resolve()

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _find_project_root(_THIS_FILE)

# -----------------------------
# Canonical directories (absolute)
# -----------------------------

user        = str(Path.home())
base_dir    = str(PROJECT_ROOT)
data_dir    = str(PROJECT_ROOT / "data")
fig_dir     = str(PROJECT_ROOT / "figures")
results_dir = str(PROJECT_ROOT / "results") 
mask_dir    = str(PROJECT_ROOT / "masks" / "ROIs")
glm_dir     = str(PROJECT_ROOT / "analyses" / "lsa")

# (optional) quick sanity checks (fail fast if layout is wrong)
for _p in (data_dir, fig_dir, mask_dir, glm_dir):
    if not Path(_p).exists():
        warnings.warn(f"[main.py] Expected path does not exist: {_p}")

# load shared utils for SNT projects
UTILS_DIR = Path.home() / "Desktop" / "shared_utils"
sys.path.insert(0, str(UTILS_DIR))
from utils_snt import *  

#-------------------------------- task info

# load the task details file (*in-person only*)
snt_df = pd.read_excel(f'{data_dir}/info/social-navigation-task.xlsx')
snt_df = snt_df[np.isfinite(snt_df['trial_num'])].sort_values(by='onset')
decision_trials = snt_df[snt_df['slide_type'] == 'Decision']
dtype_dict = {'decision_num': int, 'character_role_num': int, 'character_decision_num': int, 'onset': float}
decision_trials = decision_trials.astype(dtype_dict)
decision_trials.reset_index(inplace=True, drop=True)

# character info
character_trial_labels = decision_trials['character_role_num'].values
CHARACTER_COLORS = {
    'first':     '#d62728',  # red
    'second':    '#1f77b4',  # blue
    'assistant': '#2ca02c',  # green
    'powerful':   '#9467bd',  # purple - aka newcomb
    'boss':  '#ff7f0e',  # orange - aka hayworth
    'neutral':   '#7f7f7f',  # grey
}
CHARACTERS = list(CHARACTER_COLORS.keys())
ROLE_TO_NAME = {
    1: "first",
    2: "second",
    3: "assistant",
    4: "powerful",
    5: "boss",
    6: "neutral",
}

# diagnosis info
covariates    = ["dx", "sex", "ctq_score"]
demo_controls = ['sex', 'age_years']
coc_vars      = ['screen_utox___coc', 'coc_dur_years', 'coc_age_1st_use', 'asi_coc_pastmonth', 'stcq_mri_crave_coc']
beh_vars      = ['affil_mean_mean', 'power_mean_mean']

# questionnaire cols
sni_cols  = ['sni_network_diversity', 'sni_network_size', 'sni_embedded_networks']
lsas_cols = ['lsas_av_score', 'lsas_fear_score']
isel_cols = ['isel_scr_appraisal_subscale', 'isel_scr_tangible_subscale', 'isel_scr_selfesteem_subscale', 'isel_scr_belonging_subscale']

# diff breakdowns
dx_colors     = ['blue', 'red'] # HC v. CD
ctq_colors    = ['green', 'brown'] # when binarized
sample_colors = ['darkblue', 'darkorange', 'darkgreen']  # in-person, online, tavares

#-------------------------------- main data
# {'HC': 0, 'CD': 1}, {'M': 0, 'F': 1}

data  = pd.read_excel(f'{data_dir}/data.xlsx')
ques_data = pd.read_excel(f'{data_dir}/questionnaire_data.xlsx')
data  = data.merge(ques_data, on='sub_id', how='left')

# add in motion and filter out low memory
qc_df = pd.read_excel(f'{data_dir}/quality-control/fmriprep_mean_fd.xlsx')
data  = data.merge(qc_df[['sub_id', 'task_fd']], on='sub_id', how='left')
data  = data[data['memory_mean'] > .2].reset_index(drop=True)

incl_subs = data['sub_id'].tolist()
data['iq']  = data['wrat_standard_score']
data['ctq'] = np.where(data['ctq_score'].isna(), np.nan,(data['ctq_score'] > data['ctq_score'].median()).astype(int))
data["affil_subj_mean"] = data[[f'affil_subj_{char}' for char in CHARACTERS]].mean(axis=1)
data["power_subj_mean"] = data[[f'power_subj_{char}' for char in CHARACTERS]].mean(axis=1)

#-------------------------------- add tavares data

data_tavares = pd.read_excel(f'{data_dir}/other-samples/tavares/data.xlsx')
incl_subs_tavares = data_tavares['sub_id'].unique().tolist()
data_tavares['sex'] = data_tavares['sex'].map({'M': 0, 'F': 1})

#-------------------------------- add online data

data_online = pd.read_excel(f'{data_dir}/other-samples/online/data.xlsx')
data_online = data_online[data_online['memory_mean'] > 0.20]
incl_subs_online = data_online['sub_id'].unique().tolist()
data_online['age_years'] = data_online['demo_age']
data_online['sex'] = data_online['demo_sex_1F'].replace(2, 0) 
data_online['iq'] = data_online['iq_score']

#-------------------------------- add stuff

data['affil_mean_neutral'] = 0
data['power_mean_neutral'] = 0
data_online['affil_mean_neutral'] = 0
data_online['power_mean_neutral'] = 0
data_tavares['affil_mean_neutral'] = 0
data_tavares['power_mean_neutral'] = 0


#-------------------------------- merge various behavioral results

def merge_no_repeat(left, right, on="sub_id", how="left"):
    # drop overlapping non-key columns from right to avoid repeats
    overlap = [c for c in right.columns if c != on and c in left.columns]
    if overlap:
        right = right.drop(columns=overlap)
    out = left.merge(right, on=on, how=how)
    return out

# rt models
rt_data     = pd.read_csv(f"{results_dir}/behavior/rt-model_betas.csv")
rt_online   = pd.read_csv(f"{results_dir}/behavior/rt-model_betas_online.csv")
rt_tavares  = pd.read_csv(f"{results_dir}/behavior/rt-model_betas_tavares.csv")
data = merge_no_repeat(data, rt_data, on="sub_id", how="left")
data_online = merge_no_repeat(data_online, rt_online, on="sub_id", how="left")
data_tavares = merge_no_repeat(data_tavares, rt_tavares, on="sub_id", how="left")

# memmory confusability
mem_conf        = pd.read_csv(f"{results_dir}/behavior/mem_conf_inlab_variants.csv")
mem_conf_online = pd.read_csv(f"{results_dir}/behavior/mem_conf_online_variants.csv")
data = merge_no_repeat(mem_conf, data, on="sub_id", how="left")
data_online = merge_no_repeat(mem_conf_online, data_online, on="sub_id", how="left")

# mapping scores
mapping_scores = pd.read_csv(f"{results_dir}/behavior/mapping_scores.csv")
mapping_scores_online = pd.read_csv(f"{results_dir}/behavior/mapping_scores_online.csv")
data = merge_no_repeat(mapping_scores, data, on="sub_id", how="left")
data_online = merge_no_repeat(mapping_scores_online, data_online, on="sub_id", how="left")


#-------------------------------- code

# general helpers
def print_df(df):
    display(Markdown(df.to_markdown()))

def pickle_file(file_, filename_, protocol=4):
    with open(filename_, 'wb') as f:
        pickle.dump(file_, f, protocol=protocol)
    f.close()

def load_pickle(filename_):
    with open(filename_, 'rb') as f:
        return pickle.load(f)

def flatten_upper_tri(mat):
    """ go from symmetrical matrix to vectorized/flattened upper triangle """
    return mat[np.triu_indices(len(mat), k=1)]


# analysis helpers
def run_regression(
    X, y=None, data=None,
    popmean=0,
    scale_x=True,
    scale_y=False,
    add_intercept=True,
    robust=False,                    # if True: use RLM(HuberT); else OLS/WLS
    se_type='default',               # 'default' | 'HC0' | 'HC1' | 'HC2' | 'HC3' | 'cluster'
    cluster=None,                    # group labels (array-like) if se_type='cluster'
    weights=None,                    # optional observation weights (used with WLS)
    ci_alpha=0.05,                   # CI level for coef intervals
    dropna='any',                    # 'any' or 'all' NA policy before fit
    test_assumptions=False,          # diagnostic tests (OLS/WLS context)
    return_obj=False,                # also return the fitted model object
    verbose=False
):
    """
    Fit OLS (or WLS if `weights` is provided) or robust RLM(HuberT).
    Also supports a one-sample test when `y` is None (testing mean(X) vs `popmean`).

    Change vs prior version
    -----------------------
    When scale_x=True:
      - continuous numeric predictors (n_unique>2) are z-scored
      - binary numeric predictors (0/1 or two unique finite values) are mean-centered
    This makes the intercept directly interpretable as the grand (covariate-adjusted) mean.
    """

    # ---------- small helpers ----------
    def _clean_col(s: str) -> str:
        return s.replace('.', '_') if isinstance(s, str) else s

    def _to_frame(a, default_col='col'):
        """Make a DataFrame from array/Series/DataFrame, ensuring 2D shape."""
        if isinstance(a, pd.DataFrame):
            return a.reset_index(drop=True)
        if isinstance(a, pd.Series):
            return a.to_frame().reset_index(drop=True)
        a = np.asarray(a)
        if a.ndim == 1:
            a = a[:, None]
        return pd.DataFrame(a, columns=[default_col] if a.shape[1] == 1 else None)

    def _zscore_nan(arr):
        """NaN-robust z-score (std=0 -> NaNs unchanged)."""
        arr = np.asarray(arr, dtype=float)
        mask = np.isfinite(arr)
        if not mask.any():
            return arr
        m = np.nanmean(arr[mask])
        s = np.nanstd(arr[mask])
        if s == 0 or not np.isfinite(s):
            return arr
        out = np.full_like(arr, np.nan, dtype=float)
        out[mask] = (arr[mask] - m) / s
        return out

    def _meancenter_nan(arr):
        """NaN-robust mean-centering."""
        arr = np.asarray(arr, dtype=float)
        mask = np.isfinite(arr)
        if not mask.any():
            return arr
        m = np.nanmean(arr[mask])
        out = np.full_like(arr, np.nan, dtype=float)
        out[mask] = arr[mask] - m
        return out

    def _is_binary_numeric(s: pd.Series) -> bool:
        """True if numeric series has exactly two unique finite values (e.g., 0/1)."""
        if not pd.api.types.is_numeric_dtype(s):
            return False
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return False
        u = np.unique(v)
        return u.size == 2

    # ---------- build design: y_mat (n,1), X_mat (n,p), names ----------
    # Case A: Formula-like usage (str with '~') or named columns
    if isinstance(X, str) or (isinstance(X, list) and X and isinstance(X[0], str)):
        # A1: user passed a full formula
        if isinstance(X, str) and '~' in X:
            formula = X
            if not add_intercept and '-1' not in formula and '+ 0' not in formula:
                formula = formula + ' - 1'
                if verbose:
                    print("Removed intercept via formula:", formula)
            y_df, X_df = patsy.dmatrices(formula, data.copy(), return_type="dataframe")

        # A2: user passed column name(s); construct formula y ~ X
        else:
            pred_cols = [X] if isinstance(X, str) else list(X)
            # gather columns we need
            if y is not None:
                cols = list(set(pred_cols + [y]))
                df = data[cols].copy()
                df.rename(columns=_clean_col, inplace=True)
                y_name = _clean_col(y)
                x_names = [_clean_col(c) for c in pred_cols]
                formula = f"{y_name} ~ " + " + ".join(x_names)
                if not add_intercept:
                    formula += " - 1"
                if verbose:
                    print("Using constructed formula:", formula)
                y_df, X_df = patsy.dmatrices(formula, df, return_type="dataframe")

            else:
                # one-sample test using named column X (no predictors; design is intercept-only)
                df = data[pred_cols].copy()
                df.rename(columns=_clean_col, inplace=True)
                y_col = _clean_col(pred_cols[0])
                y_df = df[[y_col]].copy()
                y_df.rename(columns={y_col: f"against_{popmean}"}, inplace=True)
                y_df.iloc[:, 0] = y_df.iloc[:, 0] - popmean
                X_df = pd.DataFrame({'Intercept': np.ones(len(y_df))})

    # Case B: array-like usage (with or without y)
    else:
        if y is not None:
            y_df = _to_frame(y, default_col='y')
            X_df = _to_frame(X)
            # name columns nicely if unnamed
            if all(isinstance(c, int) for c in X_df.columns):
                X_df.columns = [f'x{i+1}' for i in range(X_df.shape[1])]
            if add_intercept:
                X_df = sm.add_constant(X_df, has_constant='add')
                X_df.rename(columns={'const': 'Intercept'}, inplace=True)
        else:
            # one-sample test against popmean
            y_df = _to_frame(X, default_col='obs')
            first = y_df.columns[0]
            y_df.rename(columns={first: f"against_{popmean}"}, inplace=True)
            y_df.iloc[:, 0] = y_df.iloc[:, 0] - popmean
            X_df = pd.DataFrame({'Intercept': np.ones(len(y_df))})

    # ---------- optional scaling ----------
    y_name = y_df.columns[0]
    if scale_y:
        y_df[y_name] = _zscore_nan(y_df[y_name])
        if verbose:
            print(f"Z-scored outcome: {y_name}")

    if scale_x:
        for col in X_df.columns:
            if col in ('Intercept', 'const'):
                continue
            s = X_df[col]

            # If binary numeric: mean-center (NOT z-score)
            if _is_binary_numeric(s):
                X_df[col] = _meancenter_nan(s)
                if verbose:
                    print(f"Mean-centered binary predictor: {col}")
                continue

            # Otherwise z-score continuous numeric predictors
            if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 2:
                X_df[col] = _zscore_nan(s)
                if verbose:
                    print(f"Z-scored predictor: {col}")

    # ---------- drop missing together ----------
    design_df = pd.concat([y_df, X_df], axis=1)
    design_df = design_df.dropna(how=dropna)
    y_fit = design_df[[y_name]]
    X_fit = design_df.drop(columns=[y_name])

    # ---------- fit model ----------
    try:
        if robust:
            fit_res = sm.RLM(y_fit, X_fit, M=sm.robust.norms.HuberT()).fit()
            rsq = adj_rsq = aic = bic = np.nan

        else:
            if weights is not None:
                base_model = sm.WLS(y_fit, X_fit, weights=np.asarray(weights)[:len(y_fit)])
            else:
                base_model = sm.OLS(y_fit, X_fit)

            ols_res = base_model.fit()

            if se_type == 'default':
                fit_res = ols_res
            else:
                cov_type = 'cluster' if se_type == 'cluster' else se_type
                cov_kw = {}
                if se_type == 'cluster':
                    if cluster is None:
                        raise ValueError("cluster labels must be provided when se_type='cluster'.")
                    cov_kw['groups'] = np.asarray(cluster)[:len(y_fit)]
                fit_res = ols_res.get_robustcov_results(cov_type=cov_type, **cov_kw)

            rsq = getattr(ols_res, 'rsquared', np.nan)
            adj_rsq = getattr(ols_res, 'rsquared_adj', np.nan)
            aic = getattr(ols_res, 'aic', np.nan)
            bic = getattr(ols_res, 'bic', np.nan)

        conf = fit_res.conf_int(alpha=ci_alpha)
        coef = fit_res.params
        se = fit_res.bse
        tvals = fit_res.tvalues
        pvals = fit_res.pvalues

        coef_df = pd.DataFrame({
            'x': coef.index,
            'beta': coef.values,
            'se': se.values,
            't': tvals.values,
            'p': pvals.values,
            '95%_lb': conf.iloc[:, 0].values,
            '95%_ub': conf.iloc[:, 1].values
        })

    except Exception as e:
        print(f"Error during {'RLM' if robust else 'OLS/WLS'} fitting:", e)
        return (pd.DataFrame(), None) if return_obj else pd.DataFrame()

    # ---------- optional diagnostics ----------
    if test_assumptions and not robust:
        try:
            resid = np.asarray(fit_res.resid).ravel()
            exog = np.asarray(fit_res.model.exog)
            reset_p = sm.stats.linear_reset(fit_res, use_f=True).pvalue
            print(f"RESET p-value: {reset_p:.4f}")
            bg_p = acorr_breusch_godfrey(fit_res, nlags=1, store=False)[1]
            print(f"Breusch–Godfrey p-value: {bg_p:.4f}")
            bp_p = sm.stats.het_breuschpagan(resid, exog)[1]
            print(f"Breusch–Pagan p-value: {bp_p:.4f}")
            jb_p = sm.stats.jarque_bera(resid)[1]
            print(f"Jarque–Bera p-value: {jb_p:.4f}")
            exog_names = getattr(fit_res.model, 'exog_names', [])
            vif = {
                name: variance_inflation_factor(exog, i)
                for i, name in enumerate(exog_names)
                if name not in ('Intercept', 'const')
            }
            if verbose:
                print("VIFs:", vif)
        except Exception as e:
            if verbose:
                print("Diagnostics failed:", e)

    # ---------- tidy output ----------
    X_label = " + ".join([c for c in X_fit.columns])

    p_left = []
    p_right = []
    for b, p in zip(coef_df['beta'], coef_df['p']):
        p_right.append(p/2 if b > 0 else 1 - p/2)
        p_left.append(p/2 if b < 0 else 1 - p/2)

    coef_df['X'] = 't-test' if 'against_' in y_name else X_label
    coef_df['y'] = y_name
    coef_df['dof'] = getattr(fit_res, 'df_resid', np.nan)
    coef_df['rsq'] = rsq
    coef_df['adj_rsq'] = adj_rsq
    coef_df['bic'] = np.round(bic, 2) if np.isfinite(bic) else np.nan
    coef_df['aic'] = np.round(aic, 2) if np.isfinite(aic) else np.nan
    coef_df['p_left'] = p_left
    coef_df['p_right'] = p_right

    cols = ['y','x','dof','rsq','adj_rsq','bic','aic',
            'beta','se','95%_lb','95%_ub','t','p','p_left','p_right']
    out_df = coef_df[cols].copy()
    for c in ['beta','se','95%_lb','95%_ub','t','p','p_left','p_right','dof','rsq','adj_rsq','bic','aic']:
        out_df[c] = np.round(out_df[c], 3)

    return (out_df, fit_res) if return_obj else out_df

def zscore(x):
    x = np.asarray(x, float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    return (x - mu) / (sd + 1e-12)

def regress_out(x, y):
    """
    Residualize y with respect to x.
    Returns residuals (same shape as y).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.full_like(y, np.nan)

    X = np.column_stack([np.ones(mask.sum()), x[mask]])
    beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)

    y_hat = np.full_like(y, np.nan)
    y_hat[mask] = X @ beta
    return y - y_hat

def trial_correlation_matrix(X):
    """
    Trial-by-trial pattern similarity (Pearson corr across voxels),
    after z-scoring each trial pattern across voxels.
    Returns (T,T).
    """
    def zscore_rows(X, ddof=1, eps=1e-8):
        """
        Row-wise z-score (each trial pattern across voxels).
        X: (T, V)
        """
        X = np.asarray(X, float)
        mu = np.nanmean(X, axis=1, keepdims=True)
        sd = np.nanstd(X, axis=1, ddof=ddof, keepdims=True) + eps
        return (X - mu) / sd

    Z = zscore_rows(X)
    T, V = Z.shape
    return (Z @ Z.T) / (V - 1)

def stable_seed(*items) -> int:
    """Deterministic 32-bit seed from arbitrary items (stable across runs)."""
    payload = "|".join(map(str, items)).encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


# organize/load data
def load_embeddings(
    sub_id,
    *,
    neutrals=True,
    context="in-context",  # {"in-context", "no-context"}
    llm="openai",
    on_missing="none",     # {"none","empty","raise"}
    data_root=None,
):
    """
    Load decision embeddings for a subject.

    This version is safe to call from any folder because it does NOT use
    relative paths like ../data. It uses, in order:

      1. data_root argument, if provided
      2. global data_dir, if defined
      3. global PROJECT_ROOT / "data", if defined

    Directory selection, based on sub_id AFTER stripping 'sub-':
      - If int in [1, 100]:  data/other-samples/tavares/narratives/narrative-embeddings
      - If non-int:         data/other-samples/online/narratives/narrative-embeddings
      - Otherwise:          data/narratives/narrative-embeddings

    File naming:
      sub-<id>_decisions-in-context_<llm>.npz
      sub-<id>_decisions-no-context_<llm>.npz

    Neutral handling:
      - If neutrals=False, applies drop_neutral_trials() to the loaded embedding array.

    Missing-file behavior:
      - on_missing="none"  -> return None
      - on_missing="empty" -> return np.empty((0, 0), float)
      - on_missing="raise" -> raise FileNotFoundError
    """

    if on_missing not in {"none", "empty", "raise"}:
        raise ValueError("on_missing must be one of {'none', 'empty', 'raise'}")

    if context not in {"in-context", "no-context"}:
        raise ValueError("context must be 'in-context' or 'no-context'")

    # --------------------------------------------------
    # Resolve project data directory robustly
    # --------------------------------------------------

    if data_root is not None:
        data_base = Path(data_root).expanduser().resolve()

    elif "data_dir" in globals():
        data_base = Path(globals()["data_dir"]).expanduser().resolve()

    elif "PROJECT_ROOT" in globals():
        data_base = Path(globals()["PROJECT_ROOT"]).expanduser().resolve() / "data"

    else:
        raise RuntimeError(
            "Could not resolve data directory. Define `data_dir`, define `PROJECT_ROOT`, "
            "or pass `data_root='/path/to/project/data'`."
        )

    # --------------------------------------------------
    # Subject ID helpers
    # --------------------------------------------------

    def _strip_prefix(x) -> str:
        s = str(x).strip()
        return s[4:] if s.startswith("sub-") else s

    def _parse_int(s: str):
        s = str(s).strip()
        return int(s) if s.isdigit() else None

    def _format_sid(x) -> str:
        """
        Returns normalized subject id string WITH 'sub-' prefix.
        """
        raw = _strip_prefix(x)
        iv = _parse_int(raw)

        if iv is not None:
            core = f"{iv:02d}" if 1 <= iv <= 9 else str(iv)
        else:
            core = raw

        return f"sub-{core}"

    sid = _format_sid(sub_id)

    raw = _strip_prefix(sub_id)
    iv = _parse_int(raw)

    # --------------------------------------------------
    # Select embeddings directory
    # --------------------------------------------------

    if iv is not None and 1 <= iv <= 100:
        emb_dir = (
            data_base
            / "other-samples"
            / "tavares"
            / "narratives"
            / "narrative-embeddings"
        )

    elif iv is None:
        emb_dir = (
            data_base
            / "other-samples"
            / "online"
            / "narratives"
            / "narrative-embeddings"
        )

    else:
        emb_dir = data_base / "narratives" / "narrative-embeddings"

    # --------------------------------------------------
    # File name
    # --------------------------------------------------

    ctxt_suffix = (
        "decisions-in-context"
        if context == "in-context"
        else "decisions-no-context"
    )

    fpath = emb_dir / f"{sid}_{ctxt_suffix}_{llm}.npz"

    # --------------------------------------------------
    # Missing behavior
    # --------------------------------------------------

    if not fpath.exists():
        if on_missing == "raise":
            raise FileNotFoundError(f"Embeddings not found: {fpath}")
        if on_missing == "empty":
            return np.empty((0, 0), dtype=float)
        return None

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    with np.load(fpath) as z:
        if "embedding" not in z.files:
            raise KeyError(
                f"Expected key 'embedding' in {fpath}. "
                f"Available keys: {z.files}"
            )

        arr = z["embedding"].astype(float)

    # --------------------------------------------------
    # Optionally drop neutral trials
    # --------------------------------------------------

    if not neutrals:
        arr = drop_neutral_trials(arr)

    return arr

def load_behavior(
    sub_id,
    neutrals=True,
    *,
    on_missing="none",
    data_root=None,
):
    """
    Load behavior dataframe for a given subject.

    This version is safe to call from any folder because it does NOT use
    relative paths like ../../../data. It uses, in order:

      1. data_root argument, if provided
      2. global data_dir, if defined
      3. global PROJECT_ROOT / "data", if defined

    Directory selection, based on sub_id AFTER stripping 'sub-':
      - If int in [1, 100]:  data/other-samples/tavares/preprocessed/behavior
      - If non-int:         data/other-samples/online/preprocessed/behavior
      - Otherwise:          data/preprocessed/behavior

    Subject ID normalization:
      - Always uses "sub-" prefix
      - If integer in [1, 9], zero-pads to 2 digits: 1 -> sub-01
      - Otherwise keeps original digits: 18001 -> sub-18001

    Missing-file behavior:
      - on_missing="none"  -> return None
      - on_missing="empty" -> return empty DataFrame
      - on_missing="raise" -> raise FileNotFoundError
    """

    if on_missing not in {"none", "empty", "raise"}:
        raise ValueError("on_missing must be one of {'none', 'empty', 'raise'}")

    # --------------------------------------------------
    # Resolve project data directory robustly
    # --------------------------------------------------

    if data_root is not None:
        data_base = Path(data_root).expanduser().resolve()

    elif "data_dir" in globals():
        data_base = Path(globals()["data_dir"]).expanduser().resolve()

    elif "PROJECT_ROOT" in globals():
        data_base = Path(globals()["PROJECT_ROOT"]).expanduser().resolve() / "data"

    else:
        raise RuntimeError(
            "Could not resolve data directory. Define `data_dir`, define `PROJECT_ROOT`, "
            "or pass `data_root='/path/to/project/data'`."
        )

    # --------------------------------------------------
    # Subject ID helpers
    # --------------------------------------------------

    def _strip_prefix(x) -> str:
        s = str(x).strip()
        return s[4:] if s.startswith("sub-") else s

    def _parse_int(s: str):
        s = str(s).strip()
        return int(s) if s.isdigit() else None

    def _format_sid(x) -> str:
        """
        Returns normalized subject id string WITH 'sub-' prefix.
        """
        raw = _strip_prefix(x)
        iv = _parse_int(raw)

        if iv is not None:
            core = f"{iv:02d}" if 1 <= iv <= 9 else str(iv)
        else:
            core = raw

        return f"sub-{core}"

    sid = _format_sid(sub_id)

    raw = _strip_prefix(sub_id)
    iv = _parse_int(raw)

    # --------------------------------------------------
    # Select behavior directory
    # --------------------------------------------------

    if iv is not None and 1 <= iv <= 100:
        beh_dir = data_base / "other-samples" / "tavares" / "preprocessed" / "behavior"

    elif iv is None:
        beh_dir = data_base / "other-samples" / "online" / "preprocessed" / "behavior"

    else:
        beh_dir = data_base / "preprocessed" / "behavior"

    fpath = beh_dir / f"{sid}.xlsx"

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    if not fpath.exists():
        if on_missing == "raise":
            raise FileNotFoundError(f"Behavior file not found: {fpath}")
        if on_missing == "empty":
            return pd.DataFrame()
        return None

    df = pd.read_excel(fpath)

    # --------------------------------------------------
    # Optionally drop neutral trials
    # --------------------------------------------------

    if not neutrals:
        if "character_role_num" in df.columns:
            role_col = "character_role_num"
        elif "char_role_num" in df.columns:
            role_col = "char_role_num"
        else:
            if on_missing == "raise":
                raise ValueError(
                    f"Behavior file has no role column: {fpath}. "
                    "Expected `character_role_num` or `char_role_num`."
                )
            if on_missing == "empty":
                return pd.DataFrame()
            return None

        df = df[df[role_col] != 6].reset_index(drop=True)

    return df

def load_subject_data(
    sub_id,
    *,
    atlas=None,
    glm_dir=None,
    load_sem=True,
    load_fmri=True,
    neutrals=True,
    llm="openai",
    emb_on_missing="none",
    beh_on_missing="none",
    do_dots_mapping=True,
    skip_sem_if_missing=True,
    verbose=True,
    sub_id_width=2,
):
    """
    Minimal, readable subject loader.

    Loads:
      - behavior
      - optional semantic embeddings
      - optional fMRI LSS decision betas into atlas ROIs

    Current fMRI beta format expected first:
      glm_dir/sub-18002_decision_trials_beta.nii.gz

    Older fallback formats also checked:
      glm_dir/sub-18002/beta_decisions.nii.gz
      glm_dir/sub-18002/beta_decisions_resampled.nii.gz
    """

    def log(msg: str):
        if verbose:
            print(f"[load_subject_data sub={sub_id}] {msg}")

    # -------------------------
    # Normalize subject id
    # -------------------------

    sid_raw = str(sub_id).strip()
    if sid_raw.startswith("sub-"):
        sid_raw = sid_raw[4:]

    is_numeric = sid_raw.isdigit()
    sid_int = int(sid_raw) if is_numeric else None
    is_two_digit_int = bool(is_numeric and 10 <= sid_int <= 99)
    is_main_sample = bool(is_numeric and sid_int >= 100)

    sid_str = sid_raw.zfill(sub_id_width) if is_numeric else sid_raw
    sid_with_prefix = f"sub-{sid_str}"

    out = {"sub_id": sid_str}
    log(f"sid_str={sid_str}, is_two_digit_int={is_two_digit_int}, main_sample={is_main_sample}")

    # -------------------------
    # Behavior
    # -------------------------

    try:
        behav = load_behavior(sid_str, neutrals=neutrals, on_missing=beh_on_missing)
    except Exception as e:
        raise RuntimeError(f"Behavior loading raised for sub {sid_str}") from e

    if behav is None:
        log("Behavior is None -> returning None")
        return None

    out["behavior"] = behav
    T_beh = len(behav)

    role_col = (
        "character_role_num" if "character_role_num" in behav.columns
        else "char_role_num" if "char_role_num" in behav.columns
        else None
    )

    if role_col is None and T_beh > 0:
        raise ValueError(
            f"Missing role column for sub {sid_str} "
            "(expected character_role_num or char_role_num)"
        )

    out["char_roles"] = behav[role_col].to_numpy(int) if role_col else np.array([], int)

    # -------------------------
    # Dots: always attempt to attach
    # -------------------------

    out.update(
        {
            "dots": None,
            "dots_roles": None,
            "dots_affine": None,
        }
    )

    dots_df = globals().get("data" if is_main_sample else "data_online", None)

    if isinstance(dots_df, pd.DataFrame) and "sub_id" in dots_df.columns:
        sid_col = dots_df["sub_id"].astype(str)

        row = dots_df[sid_col == str(sid_str)]
        if row.empty and is_numeric:
            row = dots_df[sid_col == str(sid_raw)]
        if row.empty:
            row = dots_df[sid_col == sid_with_prefix]

        if not row.empty:
            try:
                dots_arr = get_coords(row.iloc[0], which="dots", include_neutral=True)
                dots_by_char = np.asarray(dots_arr)[0]

                roles = ["first", "second", "assistant", "powerful", "boss", "neutral"]

                try:
                    global_roles = list(CHARACTERS)
                    if set(global_roles) == set(roles) and global_roles != roles:
                        idx = [roles.index(r) for r in global_roles]
                        dots_by_char = dots_by_char[idx]
                        roles = global_roles
                except Exception:
                    pass

                out.update(
                    {
                        "dots": dots_by_char,
                        "dots_roles": roles,
                    }
                )

                if (
                    do_dots_mapping
                    and T_beh > 0
                    and {"affil_coord", "power_coord"} <= set(behav.columns)
                ):
                    beh_xy = behav[["affil_coord", "power_coord"]].to_numpy(float)
                    beh_dots, affine = transform_beh_to_dots(
                        beh_xy,
                        dots_by_char,
                        anchor="end",
                    )
                    behav[["affil_coord_in_dots", "power_coord_in_dots"]] = beh_dots
                    out["dots_affine"] = affine

            except Exception as e:
                log(f"Dots attach/mapping skipped: {e}")
    else:
        log("No dots df found: missing `data`/`data_online` or not a DataFrame -> skipping dots.")

    # -------------------------
    # Embeddings
    # -------------------------

    out.update({"embds_ctxt": None, "embds_noctxt": None, "has_sem": False})

    if load_sem:
        emb_ctxt = None
        emb_noctxt = None

        try:
            emb_ctxt = load_embeddings(
                sid_str,
                neutrals=neutrals,
                context="in-context",
                llm=llm,
                on_missing=emb_on_missing,
            )
        except Exception as e:
            log(f"Embeddings (in-context) skipped: {e}")

        try:
            emb_noctxt = load_embeddings(
                sid_str,
                neutrals=neutrals,
                context="no-context",
                llm=llm,
                on_missing=emb_on_missing,
            )
        except Exception as e:
            log(f"Embeddings (no-context) skipped: {e}")

        if emb_ctxt is not None and emb_noctxt is not None:
            out.update(
                {
                    "embds_ctxt": emb_ctxt,
                    "embds_noctxt": emb_noctxt,
                    "has_sem": True,
                }
            )

            if T_beh > 0:
                if emb_ctxt.shape[0] != T_beh:
                    raise ValueError(
                        f"Embedding/behavior length mismatch for {sid_str}: "
                        f"in-context={emb_ctxt.shape[0]}, behavior={T_beh}"
                    )

                if emb_noctxt.shape[0] != T_beh:
                    raise ValueError(
                        f"Embedding/behavior length mismatch for {sid_str}: "
                        f"no-context={emb_noctxt.shape[0]}, behavior={T_beh}"
                    )

        elif not skip_sem_if_missing:
            log("Semantic embeddings missing and skip_sem_if_missing=False -> returning None")
            return None

    # -------------------------
    # fMRI loading switch
    # -------------------------

    if atlas is None or glm_dir is None:
        load_fmri = False

    # -------------------------
    # fMRI: atlas ROI extraction
    # -------------------------

    if load_fmri:
        import nibabel as nib
        from nilearn import image
        from sklearn.feature_selection import VarianceThreshold

        # -------------------------
        # Load / parse atlas
        # -------------------------

        if isinstance(atlas, str):
            atlas = pd.read_pickle(atlas) if atlas.endswith(".pkl") else nib.load(atlas)

        if isinstance(atlas, dict):
            atlas_img = atlas["image"]
            rois = atlas["rois"]

            if isinstance(rois, dict):
                atlas_codes = [int(k) for k in rois.keys()]
                atlas_labels = [str(v) for v in rois.values()]
            else:
                atlas_labels = list(rois)
                atlas_codes = list(range(1, len(atlas_labels) + 1))

        elif isinstance(atlas, nib.Nifti1Image):
            atlas_img = atlas
            labels = np.unique(atlas_img.get_fdata().astype(int))
            labels = labels[labels > 0]
            atlas_codes = [int(i) for i in labels]
            atlas_labels = [f"ROI-{i}" for i in labels]

        else:
            raise ValueError(f"atlas must be dict, str, or Nifti1Image; got {type(atlas)}")

        # -------------------------
        # Load beta image
        # -------------------------
        # New LSS flat format first:
        #   glm_dir/sub-18002_decision_trials_beta.nii.gz
        #
        # Then older nested fallbacks:
        #   glm_dir/sub-18002/beta_decisions.nii.gz
        #   glm_dir/sub-18002/beta_decisions_resampled.nii.gz

        beta_candidates = [
            os.path.join(glm_dir, f"{sid_with_prefix}_decision_trials_beta.nii.gz"),
            os.path.join(glm_dir, sid_with_prefix, "beta_decisions.nii.gz"),
            os.path.join(glm_dir, sid_with_prefix, "beta_decisions_resampled.nii.gz"),
        ]

        beta_path = next((p for p in beta_candidates if os.path.exists(p)), None)

        if beta_path is None:
            msg = "Beta image not found. Checked:\n" + "\n".join(f"  - {p}" for p in beta_candidates)
            raise FileNotFoundError(msg)

        log(f"Loading beta image: {beta_path}")
        out["beta_path"] = beta_path

        beta_img = nib.load(beta_path)
        beta = beta_img.get_fdata().astype(float)

        if beta.ndim != 4:
            raise ValueError(f"Expected 4D beta image, got shape {beta.shape}")

        T_fmri = beta.shape[3]

        if T_beh > 0 and T_fmri != T_beh:
            raise ValueError(
                f"fMRI/behavior length mismatch for {sid_str}: "
                f"fmri={T_fmri}, behavior={T_beh}"
            )

        # -------------------------
        # Resample atlas to beta image
        # -------------------------

        beta_flat = beta.reshape(-1, T_fmri)

        atlas_resamp = image.resample_to_img(
            atlas_img,
            beta_img,
            interpolation="nearest",
        )

        atlas_flat = atlas_resamp.get_fdata().astype(int).ravel()

        if atlas_flat.shape[0] != beta_flat.shape[0]:
            raise ValueError(
                f"Atlas/beta voxel mismatch after resampling for {sid_str}: "
                f"atlas voxels={atlas_flat.shape[0]}, beta voxels={beta_flat.shape[0]}"
            )

        # -------------------------
        # Extract ROI beta matrices
        # -------------------------

        roi_betas = {}

        for code, name in zip(atlas_codes, atlas_labels):
            mat = beta_flat[atlas_flat == int(code)].T  # shape: (T, V)

            if mat.size:
                # Drop non-finite voxels
                mat = mat[:, np.isfinite(mat).all(axis=0)]

                # Drop zero-variance voxels
                if mat.shape[1] > 0:
                    mat = VarianceThreshold(threshold=0.0).fit_transform(mat)

            roi_betas[name] = mat

        out["roi_betas"] = roi_betas
        out["has_fmri"] = True

    else:
        out["roi_betas"] = None
        out["has_fmri"] = False

    log("Success.")
    return out


# plotting helpers
def set_plot_style(
    *,
    font_family="Arial",    
    label_size=12,
    tick_size=10,
    legend_size=9,
    grid=False,
    figsize=(6, 4), dpi=300
):
    """Standardize plot aesthetics:
       - titles == axis-label sizes
       - tick labels smaller
       - consistent fonts/lines/legends
       - no top/right spines
    """
    sns.set_theme(
        context="paper",
        style="white",
        palette="deep",
        rc={
            # --- Fonts
            "font.family": font_family,
            "font.size": label_size,
            "axes.titlesize": label_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,

            # --- Ticks
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,

            # --- Grid
            "axes.grid": grid,
            "grid.color": "black",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,

            # --- Legend
            "legend.frameon": True,
            "legend.loc": "upper right",
            "legend.fontsize": legend_size,
            "legend.title_fontsize": label_size,

            # --- Figure / lines
            "figure.figsize": figsize,
            "figure.dpi": dpi,
            "axes.linewidth": 1.0,
            "lines.linewidth": 1.0,
            "lines.markersize": 6,

            # --- Spines (remove top/right)
            "axes.spines.top": False,
            "axes.spines.right": False,

            # --- Saving
            "savefig.dpi": 300,
            "savefig.format": "svg",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        },
    )

    # Matplotlib-only niceties
    mpl.rcParams["axes.titleweight"] = "normal"
    mpl.rcParams["axes.labelweight"] = "normal"

set_plot_style()

def save_plot(filename, dpi=300, fmt='png', close=False):
    """
    Save a plot with standard settings.

    Args:
        filename (str | Path): Path (can include folders) to save to.
        dpi (int): Resolution of the saved image.
        fmt (str): File format to save ('svg', 'png', etc.).
        close (bool): Whether to close the figure after saving.
    """
    path = Path(filename)

    # Ensure the desired extension
    if path.suffix.lstrip('.').lower() != fmt.lower():
        path = path.with_suffix(f'.{fmt}')

    # Create parent directories if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(path.as_posix(), dpi=dpi, format=fmt, bbox_inches='tight')
    if close:
        plt.close()

def add_pval(p, ax, x=0.5, y=0.925, fontsize=20, color="black"):
    """
    Annotate significance stars on `ax` in axes-fraction coords.

    Safely handles p=None, p=nan, or array-like p.
    """
    if p is None:
        return
    try:
        p = float(np.asarray(p).squeeze())
    except Exception:
        return
    if not np.isfinite(p):
        return

    if p < 0.001:
        stars = '***'
    elif p < 0.01:
        stars = '**'
    elif p < 0.05:
        stars = '*'
    else:
        stars = ''

    ax.annotate(
        stars,
        xy=(x, y),
        xycoords='axes fraction',
        fontsize=fontsize,
        ha='center',
        va='center',
        color=color,
    )


# general stat plots
def regress_and_plot_betas(
    df,
    outcome,
    x_columns,
    covariates=None,
    alpha=0.05,
    cov_type="HC3",
    sort_by_beta=True,
    figsize=None,
    title=None,
    color="steelblue",
    rotation=90,
    *,
    standardize=False,          # z-score outcome + predictors within each regression subset
    min_n=10,
    drop_constant=True,
    eps=1e-12,
    verbose=False,
):
    """
    Run separate OLS regressions:
        outcome ~ x + covariates
    for each x in x_columns, and plot beta coefficients with 95% CIs.

    Key robustness features:
      - Handles duplicate column names by selecting the first occurrence (records this fact).
      - Coerces to numeric via pd.to_numeric(errors="coerce") on 1D Series only.
      - Complete-case rows per model.
      - Optional within-model standardization.
      - Skips near-constant outcome/predictors; drops near-constant covariates per model.

    Returns
    -------
    results_df : pd.DataFrame
    ax : matplotlib Axes
    meta : dict (includes 'skipped', 'duplicates')
    """

    def _sig_stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple, pd.Index)):
            return list(x)
        if isinstance(x, np.ndarray):
            return list(x.tolist())
        return [x]

    def _flatten_names(seq):
        out = []
        for item in _as_list(seq):
            if isinstance(item, (list, tuple, np.ndarray, pd.Index)):
                out.extend(_flatten_names(item))
            else:
                out.append(item)
        return [str(s) for s in out]

    def _get_series_first(frame: pd.DataFrame, name: str, duplicates_log: list) -> pd.Series:
        """
        Always return a 1D Series for a given column name.
        If duplicates exist, take the first occurrence and log it.
        """
        if name not in frame.columns:
            raise KeyError(name)

        # If duplicates exist, frame[name] is a DataFrame; else Series.
        obj = frame[name]
        if isinstance(obj, pd.DataFrame):
            duplicates_log.append((name, obj.shape[1]))
            s = obj.iloc[:, 0]
        else:
            s = obj
        # ensure name is preserved
        s = s.copy()
        s.name = name
        return s

    def _coerce_numeric_series(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    def _zscore_series(s: pd.Series) -> pd.Series:
        arr = s.to_numpy(dtype=float, copy=False)
        mu = np.nanmean(arr)
        sd = np.nanstd(arr, ddof=1)
        if not np.isfinite(sd) or sd < eps:
            return pd.Series(np.nan, index=s.index, name=s.name)
        return (s - mu) / (sd + eps)

    def _is_near_constant(s: pd.Series) -> bool:
        arr = s.to_numpy(dtype=float, copy=False)
        sd = np.nanstd(arr, ddof=1)
        return (not np.isfinite(sd)) or (sd < eps)

    outcome = str(outcome)
    covariates = _flatten_names(covariates)
    x_columns = _flatten_names(x_columns)

    # basic required column presence check (by name)
    needed = [outcome] + covariates
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    results = []
    skipped = []
    duplicates = []  # list of tuples: (name, n_copies)

    for x in x_columns:
        if x not in df.columns:
            skipped.append((x, "missing_column"))
            continue

        # Build a clean per-model numeric dataframe from 1D Series only
        try:
            y = _get_series_first(df, outcome, duplicates)
            xs = _get_series_first(df, x, duplicates)
            covs = {c: _get_series_first(df, c, duplicates) for c in covariates}
        except KeyError as e:
            skipped.append((x, f"missing_column({e})"))
            continue

        reg_df = pd.DataFrame({outcome: y, x: xs, **covs})

        # Coerce to numeric (safe because each is a Series)
        for c in reg_df.columns:
            reg_df[c] = _coerce_numeric_series(reg_df[c])

        # Complete-case for this model
        reg_df = reg_df.dropna(axis=0, how="any")
        n = int(len(reg_df))
        if n < int(min_n):
            skipped.append((x, f"too_few_rows(n={n})"))
            continue

        # Optional standardization
        if standardize:
            for c in reg_df.columns:
                reg_df[c] = _zscore_series(reg_df[c])
            reg_df = reg_df.dropna(axis=0, how="any")
            n = int(len(reg_df))
            if n < int(min_n):
                skipped.append((x, f"too_few_rows_after_standardize(n={n})"))
                continue

        # Drop/skip constants
        usable_covs = covariates[:]
        if drop_constant:
            if _is_near_constant(reg_df[outcome]):
                skipped.append((x, "outcome_near_constant"))
                continue
            if _is_near_constant(reg_df[x]):
                skipped.append((x, "x_near_constant"))
                continue

            # drop near-constant covariates *for this model*
            usable_covs = [c for c in covariates if not _is_near_constant(reg_df[c])]

        # Quote names for formula safety
        terms = [f"Q('{x}')"] + [f"Q('{c}')" for c in usable_covs]
        formula = f"Q('{outcome}') ~ " + " + ".join(terms)

        try:
            model = smf.ols(formula, data=reg_df).fit(cov_type=cov_type)
        except Exception as e:
            skipped.append((x, f"fit_failed({type(e).__name__}: {e})"))
            continue

        x_term = f"Q('{x}')"
        if x_term not in model.params.index:
            skipped.append((x, "x_term_not_in_params"))
            continue

        beta = float(model.params.loc[x_term])
        pval = float(model.pvalues.loc[x_term])
        ci_lb, ci_ub = model.conf_int(alpha=alpha).loc[x_term].to_list()

        results.append(
            dict(
                questionnaire=x,
                n=n,
                beta=beta,
                ci_lb=float(ci_lb),
                ci_ub=float(ci_ub),
                pvalue=pval,
                stars=_sig_stars(pval),
                covariates_used=",".join(usable_covs),
            )
        )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise RuntimeError(
            "No models were fit successfully. "
            f"Examples of skips: {skipped[:10]}. "
            f"Duplicate-column events (first 10): {duplicates[:10]}"
        )

    if sort_by_beta:
        results_df = results_df.sort_values("beta", ascending=True).reset_index(drop=True)
    else:
        results_df = results_df.reset_index(drop=True)

    # ---- Plot ----
    if figsize is None:
        figsize = (max(10, 0.35 * len(results_df)), 8)

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=results_df,
        x="questionnaire",
        y="beta",
        ax=ax,
        color=color,
        ci=None,
    )

    y_min = float(np.nanmin(results_df["ci_lb"]))
    y_max = float(np.nanmax(results_df["ci_ub"]))
    y_range = y_max - y_min
    offset = 0.02 * (y_range if np.isfinite(y_range) and y_range > 0 else 1.0)

    for xi, row in enumerate(results_df.itertuples(index=False)):
        ax.plot([xi, xi], [row.ci_lb, row.ci_ub], color="black", lw=1)
        if row.stars:
            star_y = (row.ci_ub + offset) if row.beta >= 0 else (row.ci_lb - offset)
            ax.text(
                xi,
                star_y,
                row.stars,
                ha="center",
                va="bottom" if row.beta >= 0 else "top",
                fontsize=12,
            )

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Beta coefficient")
    ax.tick_params(axis="x", labelrotation=rotation)
    if title:
        ax.set_title(title)
    plt.tight_layout()

    meta = dict(skipped=skipped, duplicates=duplicates)
    if verbose:
        if duplicates:
            # summarize duplicates by name
            dup_counts = {}
            for name, k in duplicates:
                dup_counts[name] = max(dup_counts.get(name, 0), k)
            print("Duplicate column names detected (name -> # copies, first occurrence used):")
            for name, k in sorted(dup_counts.items(), key=lambda t: (-t[1], t[0]))[:25]:
                print(f"  {name}: {k}")
        if skipped:
            print(f"Skipped {len(skipped)} predictors. First 15:")
            for item in skipped[:15]:
                print("  ", item)

    return results_df, ax, meta

def plot_rdm(
    X,
    *,
    metric="euclidean",
    ax=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    square=True,
    cbar=True,
    cbar_kws=None,
    xticklabels=False,
    yticklabels=False,
    title=None,
):
    """
    Compute a representational dissimilarity matrix (RDM) via pairwise_distances
    and plot it as a seaborn heatmap.

    If X contains strings (or other non-numeric labels), it is first encoded to
    integer IDs (stable order of first appearance) before computing distances.
    """
    X = np.asarray(X)

    # If metric=None, X is interpreted as a precomputed distance matrix.
    if metric is None:
        D = np.asarray(X, float)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f"When metric=None, X must be a square distance matrix; got {D.shape}.")
    else:
        if X.ndim == 1:
            X = X[:, None]
        elif X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D to compute pairwise distances; got {X.shape}.")

        # Encode string/object arrays to integer IDs (per column) before distances
        if X.dtype.kind in ("U", "S", "O"):
            X_enc = np.empty(X.shape, dtype=float)
            for j in range(X.shape[1]):
                col = X[:, j]
                # stable mapping by first appearance
                mapping = {}
                next_id = 0
                ids = np.empty(col.shape[0], dtype=float)
                for i, v in enumerate(col):
                    key = v
                    if key not in mapping:
                        mapping[key] = next_id
                        next_id += 1
                    ids[i] = mapping[key]
                X_enc[:, j] = ids
            X = X_enc
        else:
            X = X.astype(float, copy=False)

        D = pairwise_distances(X, metric=metric)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.8))

    sns.heatmap(
        D,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=square,
        cbar=cbar,
        cbar_kws=cbar_kws,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
    )

    ax.set_xlabel("Trial")
    ax.set_ylabel("Trial")
    if title is not None:
        ax.set_title(title)

    return ax, D


# roi-related plots
def roi_one_sample_tests(
    df: pd.DataFrame,
    *,
    value_col: str,
    rois: Sequence[str],
    roi_col: str = "roi",
    test: Literal["ttest_1samp", "wilcoxon"] = "ttest_1samp",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ref: float = 0.0,
    min_n: int = 2,
    covariates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by ROI with columns: n, mean, p.

    If covariates is provided (non-empty), runs an ANCOVA-style OLS model:
        y ~ 1 + covariates
    (with covariates mean-centered within ROI), and tests the intercept vs ref
    using a t-test with df = n - (k + 1).

    Note: when covariates is provided, `test` is ignored.
    """
    covariates = tuple(covariates) if covariates else None

    rows = []
    for roi in rois:
        if not covariates:
            y = df.loc[df[roi_col] == roi, value_col].to_numpy(float)
            y = y[np.isfinite(y)]
            n = int(y.size)

            p = np.nan
            if n >= min_n:
                if test == "ttest_1samp":
                    res = st.ttest_1samp(y, popmean=ref, nan_policy="omit", alternative=alternative)
                    p = float(res.pvalue) if np.isfinite(res.pvalue) else np.nan
                elif test == "wilcoxon":
                    d = y - ref
                    d = d[np.isfinite(d)]
                    if d.size and (not np.allclose(d, 0.0)):
                        try:
                            res = st.wilcoxon(d, alternative=alternative)
                            p = float(res.pvalue) if np.isfinite(res.pvalue) else np.nan
                        except ValueError:
                            p = np.nan
                else:
                    raise ValueError("test must be 'ttest_1samp' or 'wilcoxon'")

            rows.append(
                dict(
                    roi=roi,
                    n=n,
                    mean=(float(np.mean(y)) if n else np.nan),
                    p=p,
                )
            )
            continue

        # ANCOVA branch
        sub = df.loc[df[roi_col] == roi, [value_col, *covariates]].copy()

        y_all = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(float)
        Xcov_all = np.column_stack(
            [pd.to_numeric(sub[c], errors="coerce").to_numpy(float) for c in covariates]
        )

        ok = np.isfinite(y_all) & np.all(np.isfinite(Xcov_all), axis=1)
        y = y_all[ok]
        Xcov = Xcov_all[ok]
        n = int(y.size)

        p = np.nan
        if n >= min_n:
            # mean-center covariates within ROI so intercept corresponds to adjusted mean
            Xcov = Xcov - Xcov.mean(axis=0, keepdims=True)
            X = np.column_stack([np.ones(n, dtype=float), Xcov])

            df_resid = n - X.shape[1]
            if df_resid > 0:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - (X @ beta)
                rss = float(np.dot(resid, resid))
                s2 = rss / float(df_resid)

                XtX_inv = np.linalg.pinv(X.T @ X)
                se0 = float(np.sqrt(max(0.0, s2 * XtX_inv[0, 0])))

                if np.isfinite(se0) and se0 > 0:
                    t = float((beta[0] - ref) / se0)
                    if np.isfinite(t):
                        if alternative == "two-sided":
                            p = float(2.0 * st.t.sf(abs(t), df=df_resid))
                        elif alternative == "greater":
                            p = float(st.t.sf(t, df=df_resid))
                        elif alternative == "less":
                            p = float(st.t.cdf(t, df=df_resid))
                        else:
                            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

        rows.append(
            dict(
                roi=roi,
                n=n,
                mean=(float(np.mean(y)) if n else np.nan),
                p=p,
            )
        )

    return pd.DataFrame(rows).set_index("roi")

def roi_pairwise_paired_ttests(
    df: pd.DataFrame,
    *,
    value_col: str,
    rois: Sequence[str],
    subject_col: str = "sub_id",
    roi_col: str = "roi",
    min_n: int = 5,
    covariates: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      t_mat: (R,R) paired t-values (row roi_i vs col roi_j for test of (i - j))
      p_mat: (R,R) two-sided p-values
      n_mat: (R,R) number of paired subjects per cell
      wide : subjects × rois pivot used for tests

    If covariates is provided (non-empty), runs an ANCOVA-style OLS model per cell:
        d_ij = (roi_i - roi_j) ~ 1 + covariates
    (with covariates mean-centered within the included subjects for that cell),
    and tests the intercept vs 0 using a two-sided t-test.
    """
    covariates = tuple(covariates) if covariates else None

    wide = (
        df.pivot_table(index=subject_col, columns=roi_col, values=value_col, aggfunc="mean")
          .reindex(columns=list(rois))
    )

    t_mat = pd.DataFrame(np.nan, index=rois, columns=rois, dtype=float)
    p_mat = pd.DataFrame(np.nan, index=rois, columns=rois, dtype=float)
    n_mat = pd.DataFrame(0,     index=rois, columns=rois, dtype=int)

    cov_wide = None
    if covariates:
        cov_wide = (
            df[[subject_col, *covariates]]
              .groupby(subject_col, as_index=True)
              .first()
              .reindex(wide.index)
              .copy()
        )
        for c in covariates:
            cov_wide[c] = pd.to_numeric(cov_wide[c], errors="coerce")

    for i, roi_i in enumerate(rois):
        xi = wide[roi_i]
        for j, roi_j in enumerate(rois):
            if i == j:
                if cov_wide is None:
                    n_mat.loc[roi_i, roi_j] = int(np.isfinite(xi).sum())
                else:
                    ok_diag = np.isfinite(xi.to_numpy(float))
                    for c in covariates:
                        ok_diag &= np.isfinite(cov_wide[c].to_numpy(float))
                    n_mat.loc[roi_i, roi_j] = int(ok_diag.sum())

                t_mat.loc[roi_i, roi_j] = 0.0
                p_mat.loc[roi_i, roi_j] = np.nan
                continue

            xj = wide[roi_j]
            ok = np.isfinite(xi) & np.isfinite(xj)

            if cov_wide is not None:
                for c in covariates:
                    ok &= np.isfinite(cov_wide[c])

            n = int(ok.sum())
            n_mat.loc[roi_i, roi_j] = n
            if n < min_n:
                continue

            if cov_wide is None:
                res = st.ttest_rel(
                    xi[ok].to_numpy(float),
                    xj[ok].to_numpy(float),
                    nan_policy="omit",
                )
                t_mat.loc[roi_i, roi_j] = float(res.statistic) if np.isfinite(res.statistic) else np.nan
                p_mat.loc[roi_i, roi_j] = float(res.pvalue)    if np.isfinite(res.pvalue)    else np.nan
                continue

            # ANCOVA on paired differences: d = xi - xj
            d = (xi[ok] - xj[ok]).to_numpy(float)
            Xcov = cov_wide.loc[ok, list(covariates)].to_numpy(float)

            # mean-center covariates within this cell
            Xcov = Xcov - Xcov.mean(axis=0, keepdims=True)
            X = np.column_stack([np.ones(n, dtype=float), Xcov])

            df_resid = n - X.shape[1]
            if df_resid <= 0:
                continue

            beta = np.linalg.lstsq(X, d, rcond=None)[0]
            resid = d - (X @ beta)
            rss = float(np.dot(resid, resid))
            s2 = rss / float(df_resid)

            XtX_inv = np.linalg.pinv(X.T @ X)
            se0 = float(np.sqrt(max(0.0, s2 * XtX_inv[0, 0])))

            if not (np.isfinite(se0) and se0 > 0):
                continue

            t = float(beta[0] / se0)
            p = float(2.0 * st.t.sf(abs(t), df=df_resid)) if np.isfinite(t) else np.nan

            t_mat.loc[roi_i, roi_j] = t
            p_mat.loc[roi_i, roi_j] = p

    return t_mat, p_mat, n_mat, wide

def plot_roi_barstrip(
    ax,
    df: pd.DataFrame,
    *,
    value_col: str,
    rois: Sequence[str],
    roi_colors: dict,
    roi_col: str = "roi",
    title: str = "",
    ylabel: str = "",
    test: Literal["ttest_1samp", "wilcoxon"] = "ttest_1samp",
    ref: float = 0.0,
    min_n: int = 2,
    jitter: float = 0.12,
    point_size: float = 4.5,
    alpha_points: float = 0.6,
    bar_alpha: float = 0.85,
    star_fontsize: int = 18,
    star_y_axes: float = 0.92,    # axes-fraction y for stars
    show_p_values: bool = False,
    p_fmt: str = "p={p:.3g}",
    p_fontsize: int = 10,
    p_y_axes: float = 0.84,       # axes-fraction y for p-text
) -> pd.DataFrame:
    """
    Returns the per-ROI stats DataFrame (n, mean, p) indexed by ROI.
    """
    d = df[[roi_col, value_col]].copy()
    d = d[np.isfinite(d[value_col].to_numpy(float))]
    d[roi_col] = pd.Categorical(d[roi_col], categories=list(rois), ordered=True)

    palette = {r: roi_colors.get(r, "gray") for r in rois}

    sns.barplot(
        data=d, x=roi_col, y=value_col, order=list(rois),
        estimator=np.mean, errorbar=None,
        palette=palette, alpha=bar_alpha,
        edgecolor="black", linewidth=0.6, ax=ax, zorder=1,
    )

    sns.stripplot(
        data=d, x=roi_col, y=value_col, order=list(rois),
        palette=palette, jitter=jitter,
        size=point_size, alpha=alpha_points,
        edgecolor="black", linewidth=0.3, ax=ax, zorder=3,
    )

    stats = roi_one_sample_tests(
        df,
        value_col=value_col,
        rois=rois,
        roi_col=roi_col,
        test=test,
        ref=ref,
        min_n=min_n,
    )

    # stars + optional p-values, in AXES FRACTION coords
    R = len(rois)
    for i, roi in enumerate(rois):
        p = stats.loc[roi, "p"]
        xfrac = (i + 0.5) / R

        add_pval(p, ax, x=xfrac, y=star_y_axes, fontsize=star_fontsize)

        if show_p_values and np.isfinite(p):
            ax.annotate(
                p_fmt.format(p=p),
                xy=(xfrac, p_y_axes),
                xycoords="axes fraction",
                ha="center",
                va="center",
                fontsize=p_fontsize,
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.axhline(ref, color="black", lw=0.8, alpha=0.35)
    ax.tick_params(axis="x", rotation=60)
    for t in ax.get_xticklabels():
        t.set_ha("right")

    return stats

def plot_roi_pairwise_heatmap(
    ax,
    df: pd.DataFrame,
    *,
    value_col: str,
    rois: Sequence[str],
    subject_col: str = "sub_id",
    roi_col: str = "roi",
    title: str = "",
    min_n: int = 5,
    vmax: float | None = None,
    star_fontsize: int = 18,
    cbar: bool = True,
    cbar_ax=None,
    cbar_label: str = "paired t (row − col)",
):
    """
    Heatmap of paired t-values (row ROI minus col ROI), with significance stars
    overlaid USING add_pval() in axes-fraction coordinates.
    """
    t_mat, p_mat, n_mat, wide = roi_pairwise_paired_ttests(
        df,
        value_col=value_col,
        rois=rois,
        subject_col=subject_col,
        roi_col=roi_col,
        min_n=min_n,
    )

    T = t_mat.to_numpy(float)
    if vmax is None:
        vmax = float(np.nanmax(np.abs(T))) if np.isfinite(T).any() else 1.0
        vmax = max(vmax, 1e-6)

    sns.heatmap(
        t_mat,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        linecolor="white",
        annot=False,
        cbar=cbar,
        cbar_ax=cbar_ax,
        cbar_kws={"label": cbar_label} if cbar else None,
    )

    # overlay stars via add_pval in axes-fraction space
    # note: seaborn heatmap uses a simple linear mapping: cell centers are evenly spaced.
    R = len(rois)
    for i, r in enumerate(rois):
        for j, c in enumerate(rois):
            if i == j:
                continue
            p = p_mat.loc[r, c]
            if not np.isfinite(p):
                continue
            # axes-fraction: x increases left->right; y increases bottom->top
            xfrac = (j + 0.5) / R
            yfrac = 1.0 - (i + 0.5) / R
            add_pval(p, ax, x=xfrac, y=yfrac, fontsize=star_fontsize)

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=60)
    for t in ax.get_xticklabels():
        t.set_ha("right")

    return t_mat, p_mat, n_mat, wide

def plot_roi_summary_row(
    df: pd.DataFrame,
    *,
    bar_value_col: str,
    heatmap_value_col: str,
    rois: Sequence[str],
    roi_colors: dict,
    subject_col: str = "sub_id",
    roi_col: str = "roi",
    bar_title: str = "",
    bar_ylabel: str = "",
    heatmap_title: str = "",
    figsize=(14, 5),
    # bar options (passed through)
    bar_test: Literal["ttest_1samp", "wilcoxon"] = "ttest_1samp",
    bar_ref: float = 0.0,
    # heatmap options
    min_n_pairwise: int = 5,
):
    """
    Single row: left = bar+strip+stars, right = paired ROI×ROI heatmap + shared cbar.
    Returns fig, (ax_bar, ax_hm), and stats outputs.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.05), wspace=0.25)

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_hm  = fig.add_subplot(gs[0, 1])
    cax    = fig.add_subplot(gs[0, 2])

    bar_stats = plot_roi_barstrip(
        ax_bar,
        df,
        value_col=bar_value_col,
        rois=rois,
        roi_colors=roi_colors,
        roi_col=roi_col,
        title=bar_title,
        ylabel=bar_ylabel,
        test=bar_test,
        ref=bar_ref,
    )

    t_mat, p_mat, n_mat, wide = plot_roi_pairwise_heatmap(
        ax_hm,
        df,
        value_col=heatmap_value_col,
        rois=rois,
        subject_col=subject_col,
        roi_col=roi_col,
        title=heatmap_title,
        min_n=min_n_pairwise,
        cbar=True,
        cbar_ax=cax,
    )

    return fig, (ax_bar, ax_hm), dict(
        bar_stats=bar_stats,
        heatmap_t=t_mat,
        heatmap_p=p_mat,
        heatmap_n=n_mat,
        heatmap_wide=wide,
    )

# generic plots
def histplot(
    data: pd.DataFrame,
    *,
    x,
    covariates=None,
    residualize: bool = True,
    require_covariates: bool = False,
    tail: str = "greater",           # {"two-sided","greater","less"}
    ax=None,
    bins=30,
    stat="density",
    kde=True,
    alpha=0.25,
    edgecolor=None,
    color=None,                      # default: black fill
    line_color=None,                 # default: match fill color
    line_kws=None,                   # kde line styling
    vline0: bool = True,
    mean_lines: bool = False,
    mean_linestyle: str = "--",
    mean_lw: float = 1.2,
    mean_alpha: float = 0.95,
    annotate: bool = True,
    popmean: float = 0.0,
    pval_x: float = 0.5,
    pval_y: float = 0.93,
    pval_fontsize: float = 25,
    set_xlim="auto",                 # None | False | True | "auto" | (lo, hi)
    pad_frac: float = 0.05,
    return_xlim: bool = False,
    **kwargs
):
    """
    Single-dataset histplot analogous to `histplot_samples()`.

    Behavior:
      1) Optionally residualize x on covariates (preserving mean)
      2) Plot seaborn histplot + optional KDE
      3) Optionally annotate 1-sample test vs `popmean` with stars
      4) Optional xlim: "auto" (via auto_xlim) or explicit (lo, hi)

    Assumes these helpers exist (as in your template / other plotting fns):
      - _as_list
      - _prep_covs
      - residualize_preserve_mean
      - auto_xlim
      - (optional) pval_1samp

    Notes:
      - If `pval_1samp` is not defined, falls back to scipy.stats.ttest_1samp
        with a standard one-sided conversion.
      - If `x` is not a string, it is treated as array-like values (no residualization).
    """
    if ax is None:
        ax = plt.gca()

    # ---- colors (fill + kde line)
    c_fill = "black" if color is None else color
    c_line = c_fill if line_color is None else line_color

    # ---- default kde line styling
    if line_kws is None:
        line_kws = dict(linewidth=2.5, alpha=0.95)
    else:
        line_kws = dict(line_kws)  # avoid mutating caller dict

    # Force KDE line color unless caller already set it AND no explicit line_color was passed
    if line_color is not None or ("color" not in line_kws):
        line_kws["color"] = c_line

    covariates = _as_list(covariates)

    # ---- get x array (residualized if possible)
    x_is_colname = isinstance(x, str)
    used_covs = []
    do_resid = False

    if data is None:
        x_arr = np.asarray([], float)
        xlabel = x if x_is_colname else "x"
    else:
        if x_is_colname:
            used_covs = (
                _prep_covs(
                    data, covariates,
                    require_covariates=require_covariates,
                    label="Data"
                )
                if (residualize and covariates) else []
            )
            do_resid = bool(residualize and used_covs)

            df = data.copy()
            if do_resid:
                x_vals = residualize_preserve_mean(df, x, used_covs)
                xlabel = f"{x} (adj.)"
            else:
                x_vals = pd.to_numeric(df[x], errors="coerce")
                xlabel = x

            x_arr = np.asarray(x_vals, float)
        else:
            # array-like input (no residualization)
            try:
                # if your codebase has to_array, use it
                to_array_fn = globals().get("to_array", None)
                if to_array_fn is not None:
                    x_arr = np.asarray(to_array_fn(x), float)
                else:
                    x_arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
            except Exception:
                x_arr = np.asarray(x, float)
            xlabel = "x"

    x_arr = x_arr[np.isfinite(x_arr)]

    # ---- xlim handling (mirror your regplot style + accept histplot_samples-style True)
    xlim = None
    _set_xlim = set_xlim
    if _set_xlim is True:
        _set_xlim = "auto"
    if _set_xlim in (None, False):
        pass
    elif _set_xlim == "auto":
        if x_arr.size:
            auto_xlim_fn = globals().get("auto_xlim", None)
            if auto_xlim_fn is not None:
                xlim = auto_xlim_fn([x_arr], pad_frac=pad_frac)
            else:
                # fallback auto xlim
                lo, hi = float(np.min(x_arr)), float(np.max(x_arr))
                if np.isfinite(lo) and np.isfinite(hi):
                    if lo == hi:
                        pad = 1.0 if lo == 0 else abs(lo) * pad_frac
                    else:
                        pad = (hi - lo) * pad_frac
                    xlim = (lo - pad, hi + pad)
        if xlim is not None:
            ax.set_xlim(*xlim)
    elif isinstance(_set_xlim, (tuple, list)) and len(_set_xlim) == 2:
        xlim = (float(_set_xlim[0]), float(_set_xlim[1]))
        ax.set_xlim(*xlim)

    # ---- plot
    if x_arr.size:
        sns.histplot(
            x=x_arr,
            bins=bins,
            stat=stat,
            kde=kde,
            alpha=alpha,
            edgecolor=edgecolor,
            color=c_fill,
            line_kws=line_kws,
            ax=ax,
            **kwargs
        )

    # ---- optional mean line
    if mean_lines and x_arr.size:
        ax.axvline(
            float(np.mean(x_arr)),
            color=c_line,
            linestyle=mean_linestyle,
            lw=mean_lw,
            alpha=mean_alpha
        )

    # ---- vline at 0
    if vline0:
        ax.axvline(0, color="black", lw=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(stat)

    # ---- annotation (1-sample p-value -> stars)
    if annotate and x_arr.size:
        # Prefer your helper if it exists
        pval_1samp_fn = globals().get("pval_1samp", None)

        if pval_1samp_fn is not None:
            p = pval_1samp_fn(x_arr, popmean=popmean, tail=tail)
        else:
            # fallback: scipy ttest_1samp (two-sided) then convert to one-sided if requested
            from scipy.stats import ttest_1samp

            t_stat, p_two = ttest_1samp(x_arr, popmean=popmean, nan_policy="omit")

            if tail == "two-sided":
                p = float(p_two)
            elif tail == "greater":
                # H1: mean > popmean
                p = float(p_two / 2.0) if t_stat > 0 else float(1.0 - p_two / 2.0)
            elif tail == "less":
                # H1: mean < popmean
                p = float(p_two / 2.0) if t_stat < 0 else float(1.0 - p_two / 2.0)
            else:
                raise ValueError(f"Invalid tail={tail!r}. Use 'two-sided','greater','less'.")

        def _p_to_stars(p_):
            if p_ is None or (isinstance(p_, float) and not np.isfinite(p_)):
                return ""
            if p_ < 1e-4:
                return "****"
            if p_ < 1e-3:
                return "***"
            if p_ < 1e-2:
                return "**"
            if p_ < 5e-2:
                return "*"
            return ""

        stars = _p_to_stars(p)
        ax.text(
            pval_x, pval_y,
            stars,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=pval_fontsize,
            color=c_line
        )

    return (ax, xlim) if return_xlim else ax

def regplot(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    covariates=None,
    residualize: bool = True,
    tail: str = "two-sided",          # {"two-sided","greater","less"} (passed to pval_beta_x)
    require_covariates: bool = False,
    ax=None,
    set_xlim=None,                    # None | "auto" | (lo, hi)
    pad_frac: float = 0.05,
    annotate: bool = True,
    pval_x: float = 0.5,
    pval_y: float = 0.96,
    pval_fontsize: float = 25,
    point_color=None,                 # default: black; if provided, overrides point color
    line_color=None,                  # default: match point_color (or black)
    scatter_kws={'s': 35, 'alpha': 0.75},
    line_kws=None,
    vline0: bool = False,
):
    """
    Single-dataset regplot that:
      1) runs y ~ x + covariates (+ intercept) and annotates significance of x
      2) plots asterisks if x is significant (based on p-value from pval_beta_x)
      3) plots points as black by default (optional override via point_color)
      4) plots residualized points (x and y residualized on covariates, preserving mean)

    Assumes these helpers exist (as in your template):
      - _as_list
      - _prep_covs
      - residualize_preserve_mean
      - auto_xlim
      - pval_beta_x
    """
    if ax is None:
        ax = plt.gca()

    scatter_kws = scatter_kws or dict(alpha=0.25, s=18)
    line_kws    = line_kws    or dict(alpha=0.95, linewidth=2.5)

    # colors: points black by default; line matches points unless overridden
    c_pts  = "black" if point_color is None else point_color
    c_line = c_pts if line_color is None else line_color

    covariates = _as_list(covariates)

    used_covs = (
        _prep_covs(data, covariates, require_covariates=require_covariates, label="Data")
        if (data is not None and covariates) else []
    )

    do_resid = bool(residualize and used_covs)

    # --- create plotting columns
    df = data.copy()

    if do_resid:
        df["_x_plot"] = residualize_preserve_mean(df, x, used_covs)
        df["_y_plot"] = residualize_preserve_mean(df, y, used_covs)
        x_plot, y_plot = "_x_plot", "_y_plot"
        xlab = f"{x} (adj.)"
        ylab = f"{y} (adj.)"
    else:
        df["_x_plot"] = pd.to_numeric(df[x], errors="coerce")
        df["_y_plot"] = pd.to_numeric(df[y], errors="coerce")
        x_plot, y_plot = "_x_plot", "_y_plot"  # keep consistent
        xlab = x
        ylab = y

    # --- optional xlim
    if set_xlim == "auto":
        arr = pd.to_numeric(df[x_plot], errors="coerce").to_numpy(float)
        xlim = auto_xlim([arr], pad_frac=pad_frac)
        if xlim is not None:
            ax.set_xlim(*xlim)
    elif isinstance(set_xlim, (tuple, list)) and len(set_xlim) == 2:
        ax.set_xlim(float(set_xlim[0]), float(set_xlim[1]))

    # --- plot (residualized if do_resid else raw numeric)
    sns.regplot(
        data=df,
        x=x_plot,
        y=y_plot,
        ax=ax,
        scatter_kws={**scatter_kws, "color": c_pts},
        line_kws={**line_kws, "color": c_line},
        truncate=False,
    )

    if vline0:
        ax.axvline(0, color="black", lw=1)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # --- regression p-value uses ORIGINAL columns + covariates (not residualized cols)
    if annotate:
        p = pval_beta_x(df, x=x, y=y, covariates=used_covs, tail=tail)

        def _p_to_stars(p_):
            if p_ is None or (isinstance(p_, float) and not np.isfinite(p_)):
                return ""
            if p_ < 1e-4:
                return "****"
            if p_ < 1e-3:
                return "***"
            if p_ < 1e-2:
                return "**"
            if p_ < 5e-2:
                return "*"
            return ""

        stars = _p_to_stars(p)
        # Put stars only (as requested). If you also want numeric p, change label below.
        ax.text(
            pval_x, pval_y,
            stars,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=pval_fontsize,
            color=c_line,
        )

    return ax

# plotting helpers
def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, pd.Index)):
        return list(x)
    return [x]

def _prep_covs(df, covariates, *, require_covariates=False, label="Data"):
    covariates = _as_list(covariates)
    missing = [c for c in covariates if c not in df.columns]
    used = [c for c in covariates if c in df.columns]
    if missing and require_covariates:
        raise KeyError(f"{label} is missing covariate columns: {missing}")
    return used

def _numeric_dropna(df, cols):
    d = df[list(cols)].copy()
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    return d

def to_array(x_spec, df=None):
    if isinstance(x_spec, str):
        if df is None:
            raise ValueError("If x_spec is a column name, you must pass df.")
        arr = pd.to_numeric(df[x_spec], errors="coerce").to_numpy(float)
    else:
        arr = np.asarray(x_spec, float)
    return arr[np.isfinite(arr)]

def auto_xlim(arrays, *, pad_frac=0.05):
    arrs = []
    for a in arrays:
        if a is None:
            continue
        v = np.asarray(a, float)
        v = v[np.isfinite(v)]
        if v.size:
            arrs.append(v)
    if not arrs:
        return None
    allv = np.concatenate(arrs)
    xmin, xmax = float(np.min(allv)), float(np.max(allv))
    pad = pad_frac * (xmax - xmin + 1e-12)
    return (xmin - pad, xmax + pad)

def tail_p_from_t(tval, df_resid, tail):
    if df_resid <= 0 or not np.isfinite(tval):
        return np.nan
    if tail == "two-sided":
        return float(2.0 * st.t.sf(np.abs(tval), df_resid))
    if tail == "greater":
        return float(st.t.sf(tval, df_resid))
    if tail == "less":
        return float(st.t.cdf(tval, df_resid))
    raise ValueError("tail must be 'greater', 'less', or 'two-sided'.")

def residualize_preserve_mean(df, target, covariates):
    covariates = _as_list(covariates)
    out = pd.Series(np.nan, index=df.index, dtype=float)

    cols = [target] + covariates if covariates else [target]
    d = _numeric_dropna(df, cols)
    if d.empty:
        return out

    yv = d[target].to_numpy(float)
    y_mean = float(np.mean(yv))

    if not covariates:
        out.loc[d.index] = yv
        return out

    k_params = 1 + len(covariates)
    if d.shape[0] <= k_params:
        return out

    X = sm.add_constant(d[covariates].to_numpy(float), has_constant="add")
    m = sm.OLS(yv, X).fit()
    out.loc[d.index] = (yv - m.fittedvalues) + y_mean
    return out

def residualize_preserve_mean_by(df, target, covariates, *, by, groups=None):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if groups is None:
        groups = [g for g in pd.unique(df[by]) if pd.notna(g)]

    for g in groups:
        idx = df[by] == g
        if idx.any():
            out.loc[idx] = residualize_preserve_mean(
                df.loc[idx], target, covariates
            ).loc[idx]
    return out

def pval_1samp(x, *, popmean=0.0, tail="two-sided"):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    res = st.ttest_1samp(x, popmean=popmean, nan_policy="omit")
    return tail_p_from_t(float(res.statistic), float(x.size - 1), tail)

def pval_t_ind(x0, x1, *, tail="two-sided", equal_var=False):
    x0 = np.asarray(x0, float)[np.isfinite(x0)]
    x1 = np.asarray(x1, float)[np.isfinite(x1)]
    if x0.size < 2 or x1.size < 2:
        return np.nan

    res = st.ttest_ind(x0, x1, equal_var=equal_var, nan_policy="omit")
    tval = float(res.statistic)

    if equal_var:
        df_resid = float(x0.size + x1.size - 2)
    else:
        v0, v1 = np.var(x0, ddof=1), np.var(x1, ddof=1)
        n0, n1 = float(x0.size), float(x1.size)
        num = (v0 / n0 + v1 / n1) ** 2
        den = (v0**2) / (n0**2 * (n0 - 1.0)) + (v1**2) / (n1**2 * (n1 - 1.0))
        df_resid = num / den if den > 0 else np.nan

    return tail_p_from_t(tval, df_resid, tail)

def pval_beta_x(df, *, x, y, covariates=(), tail="two-sided"):
    covariates = _as_list(covariates)
    d = _numeric_dropna(df, [y, x] + covariates)

    k_params = 1 + 1 + len(covariates)
    if d.shape[0] <= k_params:
        return np.nan

    Y = d[y].to_numpy(float)
    X = sm.add_constant(d[[x] + covariates].to_numpy(float), has_constant="add")
    m = sm.OLS(Y, X).fit()

    return tail_p_from_t(float(m.tvalues[1]), float(m.df_resid), tail)

# sample plots
def histplot_samples(
    data=None,
    data_online=None,
    data_tavares=None,
    *,
    x,
    x_online=None,
    x_tavares=None,
    covariates=None,
    residualize=True,
    require_covariates=False,
    tail="greater",
    ax=None,
    bins=30,
    stat="density",
    kde=True,
    alpha=0.25,
    edgecolor=None,
    line_kws=None,
    vline0=True,
    mean_lines=False,
    mean_linestyle="--",
    mean_lw=1.2,
    mean_alpha=0.95,
    annotate=True,
    popmean=0.0,
    pval_x=0.5,
    pval_y=0.93,
    pval_y_online=0.84,
    pval_y_tavares=0.75,
    pval_fontsize=25,
    set_xlim=True,
    pad_frac=0.05,
    return_xlim=False,
    **kwargs
):
    if ax is None:
        ax = plt.gca()

    covariates = _as_list(covariates)
    if line_kws is None:
        line_kws = dict(linewidth=2.5, alpha=0.95)

    sc = globals().get("sample_colors", ["C0", "C1", "C2"])
    c_inlab = sc[0] if data is not None else None
    c_on    = sc[1] if data_online is not None else None
    c_tav   = sc[2] if data_tavares is not None else None

    x_online  = x if x_online  is None else x_online
    x_tavares = x if x_tavares is None else x_tavares

    def get_x(df, xcol, label):
        if df is None:
            return np.asarray([], float)
        if not isinstance(xcol, str):
            return to_array(xcol)
        used = (
            _prep_covs(
                df, covariates,
                require_covariates=require_covariates,
                label=label
            )
            if residualize and covariates else []
        )
        if residualize and used:
            arr = residualize_preserve_mean(df, xcol, used).to_numpy(float)
        else:
            arr = pd.to_numeric(df[xcol], errors="coerce").to_numpy(float)
        return arr[np.isfinite(arr)]

    x_arr = get_x(data,        x,         "In-lab data")
    x_on  = get_x(data_online, x_online,  "Online data")
    x_tav = get_x(data_tavares,x_tavares, "Tavares data")

    xlabel = f"{x} (adj.)" if residualize and covariates else x
    arrays = [a for a in (x_arr, x_on, x_tav) if a.size]
    xlim = auto_xlim(arrays, pad_frac=pad_frac) if (set_xlim and arrays) else None

    if x_arr.size:
        sns.histplot(x=x_arr, color=c_inlab, **dict(
            bins=10, stat=stat, kde=kde, alpha=alpha,
            edgecolor=edgecolor, line_kws=line_kws, ax=ax, **kwargs
        ))
    if x_on.size:
        sns.histplot(x=x_on, color=c_on, **dict(
            bins=30, stat=stat, kde=kde, alpha=alpha,
            edgecolor=edgecolor, line_kws=line_kws, ax=ax, **kwargs
        ))
    if x_tav.size:
        sns.histplot(x=x_tav, color=c_tav, **dict(
            bins=5, stat=stat, kde=kde, alpha=alpha,
            edgecolor=edgecolor, line_kws=line_kws, ax=ax, **kwargs
        ))

    if mean_lines:
        if x_arr.size: ax.axvline(x_arr.mean(), color=c_inlab, linestyle=mean_linestyle, lw=mean_lw, alpha=mean_alpha)
        if x_on.size:  ax.axvline(x_on.mean(),  color=c_on,    linestyle=mean_linestyle, lw=mean_lw, alpha=mean_alpha)
        if x_tav.size: ax.axvline(x_tav.mean(), color=c_tav,   linestyle=mean_linestyle, lw=mean_lw, alpha=mean_alpha)

    if vline0:
        ax.axvline(0, color="black", lw=1)
    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(stat)

    if annotate:
        if x_arr.size:
            add_pval(
                pval_1samp(x_arr, popmean=popmean, tail=tail),
                ax, x=pval_x, y=pval_y, fontsize=pval_fontsize, color=c_inlab
            )
        if x_on.size:
            add_pval(
                pval_1samp(x_on, popmean=popmean, tail=tail),
                ax, x=pval_x, y=pval_y_online, fontsize=pval_fontsize, color=c_on
            )
        if x_tav.size:
            add_pval(
                pval_1samp(x_tav, popmean=popmean, tail=tail),
                ax, x=pval_x, y=pval_y_tavares, fontsize=pval_fontsize, color=c_tav
            )

    handles = []
    if x_arr.size: handles.append(Line2D([0],[0], color=c_inlab, lw=2.5, label="In-lab"))
    if x_on.size:  handles.append(Line2D([0],[0], color=c_on,    lw=2.5, label="Online"))
    if x_tav.size: handles.append(Line2D([0],[0], color=c_tav,   lw=2.5, label="Tavares"))

    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)

    return (ax, xlim) if return_xlim else ax

def regplot_samples(
    data=None,
    data_online=None,
    data_tavares=None,
    *,
    x,
    y,
    covariates=None,
    residualize=True,
    tail="greater",
    require_covariates=False,
    ax=None,
    set_xlim=None,
    pad_frac=0.05,
    annotate=True,
    pval_x=0.5,
    pval_y_inlab=0.96,
    pval_y_online=0.88,
    pval_y_tavares=0.80,
    pval_fontsize=25,
    scatter_kws=None,
    line_kws=None,
    vline0=False,
):
    if ax is None:
        ax = plt.gca()

    scatter_kws = scatter_kws or dict(alpha=0.25, s=18)
    line_kws    = line_kws    or dict(alpha=0.95, linewidth=2.5)

    covariates = _as_list(covariates)

    c_inlab = sample_colors[0] if data is not None else None
    c_on    = sample_colors[1] if data_online is not None else None
    c_tav   = sample_colors[2] if data_tavares is not None else None

    # --- figure out which datasets actually have covariates available
    used_inlab = (
        _prep_covs(data, covariates, require_covariates=require_covariates, label="In-lab data")
        if (data is not None and covariates) else []
    )
    used_on = (
        _prep_covs(data_online, covariates, require_covariates=require_covariates, label="Online data")
        if (data_online is not None and covariates) else []
    )
    used_tav = (
        _prep_covs(data_tavares, covariates, require_covariates=require_covariates, label="Tavares data")
        if (data_tavares is not None and covariates) else []
    )

    # Residualize if requested AND at least one dataset can be residualized.
    do_resid = residualize and bool(used_inlab or used_on or used_tav)

    def _ensure_plot_cols(df, used_covs):
        """Create _x_plot/_y_plot for a dataset (residualized if possible; else raw numeric)."""
        if df is None:
            return None
        out = df.copy()

        if do_resid:
            if used_covs:
                out["_x_plot"] = residualize_preserve_mean(out, x, used_covs)
                out["_y_plot"] = residualize_preserve_mean(out, y, used_covs)
            else:
                # No covs for this dataset: still create columns so plotting is consistent
                out["_x_plot"] = pd.to_numeric(out[x], errors="coerce")
                out["_y_plot"] = pd.to_numeric(out[y], errors="coerce")

        return out

    data_p = _ensure_plot_cols(data, used_inlab)
    on_p   = _ensure_plot_cols(data_online, used_on)
    tav_p  = _ensure_plot_cols(data_tavares, used_tav)

    x_plot = "_x_plot" if do_resid else x
    y_plot = "_y_plot" if do_resid else y
    xlab   = f"{x} (adj.)" if do_resid else x
    ylab   = f"{y} (adj.)" if do_resid else y

    # --- optional xlim handling (only if you actually use it elsewhere)
    if set_xlim == "auto":
        arrs = []
        for df_ in (data_p, on_p, tav_p):
            if df_ is None:
                continue
            arrs.append(pd.to_numeric(df_[x_plot], errors="coerce").to_numpy(float))
        xlim = auto_xlim(arrs, pad_frac=pad_frac)
        if xlim is not None:
            ax.set_xlim(*xlim)
    elif isinstance(set_xlim, (tuple, list)) and len(set_xlim) == 2:
        ax.set_xlim(float(set_xlim[0]), float(set_xlim[1]))

    # --- plot (online first, then in-lab, then tavares)
    if on_p is not None:
        sns.regplot(
            data=on_p, x=x_plot, y=y_plot, color=c_on,
            scatter_kws=scatter_kws, line_kws=line_kws, ax=ax
        )

    if data_p is not None:
        sns.regplot(
            data=data_p, x=x_plot, y=y_plot, color=c_inlab,
            scatter_kws=scatter_kws, line_kws=line_kws, ax=ax
        )

    if tav_p is not None:
        sns.regplot(
            data=tav_p, x=x_plot, y=y_plot, color=c_tav,
            scatter_kws=scatter_kws, line_kws=line_kws, ax=ax
        )

    if vline0:
        ax.axvline(0, color="black", lw=1)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # --- p-values: use ORIGINAL columns + covariates (not residualized columns)
    if annotate:
        if data is not None:
            add_pval(
                pval_beta_x(data, x=x, y=y, covariates=used_inlab, tail=tail),
                ax, pval_x, pval_y_inlab, pval_fontsize, c_inlab
            )
        if data_online is not None:
            add_pval(
                pval_beta_x(data_online, x=x, y=y, covariates=used_on, tail=tail),
                ax, pval_x, pval_y_online, pval_fontsize, c_on
            )
        if data_tavares is not None:
            add_pval(
                pval_beta_x(data_tavares, x=x, y=y, covariates=used_tav, tail=tail),
                ax, pval_x, pval_y_tavares, pval_fontsize, c_tav
            )

    # --- legend
    handles = []
    if data is not None:
        handles.append(Line2D([0],[0], color=c_inlab, lw=2.5, label="In-lab"))
    if data_online is not None:
        handles.append(Line2D([0],[0], color=c_on, lw=2.5, label="Online"))
    if data_tavares is not None:
        handles.append(Line2D([0],[0], color=c_tav, lw=2.5, label="Tavares"))

    if handles:
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)

    return ax

def barstrip_samples(
    data=None,
    data_online=None,
    data_tavares=None,
    *,
    x,
    x_online=None,
    x_tavares=None,
    covariates=None,
    residualize=True,
    require_covariates=False,
    tail="greater",
    ax=None,
    order=None,               # order of variables on x-axis
    sample_order=None,        # order of samples in hue (left->right within each variable)
    estimator=np.mean,
    errorbar=("se", 1.0),     # seaborn>=0.12; set None to disable
    capsize=0.0,
    saturation=1.0,
    bar_alpha=0.75,
    bar_edgecolor=None,
    strip=True,
    strip_size=4,
    strip_alpha=0.25,
    strip_jitter=0.18,
    strip_linewidth=0.0,
    annotate=True,
    popmean=0.0,
    pval_fontsize=8,
    pval_color=None,          # None -> match sample color
    pval_y_pad_frac=0.1,      # fraction of y-range for star offset (used for top margin)
    pval_top=True,            # NEW: place all p-value text near top uniformly
    pval_top_frac=0.95,       # NEW: top anchor as fraction of y-range (data coords)
    pval_top_dy_frac=0.06,    # NEW: vertical spacing between samples at top (fraction of y-range)
    pval_show_ns=True,        # NEW: if False, suppress "n.s."
    rotation=0,
    legend=True,
    legend_loc="upper right",
    legend_fontsize=10,
    despine=True,
    ylabel="value",
    debug=False,
    **kwargs
):
    """
    Barplot (mean +/- errorbar) with stripplot overlay for up to three samples.
    If `x` is a list, variables are grouped on x-axis and samples are hue.

    Robust p-value star placement:
      - Computes p-values per (variable, sample) as 1-sample test vs `popmean` with `tail`.
      - Places stars on the correct bar using geometry (xticks + dodge offsets), NOT patch index,
        NOT BarContainer labels, NOT color matching.

    NEW: uniform p-value placement near the top of the plot (pval_top=True).
    """

    if ax is None:
        ax = plt.gca()

    covariates = _as_list(covariates)

    # -------- sample definitions --------

    sc = globals().get("sample_colors", ["C0", "C1", "C2"])
    sample_defs = []
    if data is not None:
        sample_defs.append(("In-lab", data, sc[0]))
    if data_online is not None:
        sample_defs.append(("Online", data_online, sc[1]))
    if data_tavares is not None:
        sample_defs.append(("Tavares", data_tavares, sc[2]))

    if not sample_defs:
        ax.set_axis_off()
        return ax

    # -------- normalize x / x_online / x_tavares to lists --------

    xs = _as_list(x)
    x_online  = xs if x_online  is None else _as_list(x_online)
    x_tavares = xs if x_tavares is None else _as_list(x_tavares)

    # broadcast if needed
    if len(xs) > 1:
        if len(x_online) == 1:
            x_online = x_online * len(xs)
        if len(x_tavares) == 1:
            x_tavares = x_tavares * len(xs)

    # variable mapping per sample
    x_by_sample = {}
    for sname, _, _ in sample_defs:
        if sname == "In-lab":
            x_by_sample[sname] = xs
        elif sname == "Online":
            x_by_sample[sname] = x_online
        elif sname == "Tavares":
            x_by_sample[sname] = x_tavares

    # Canonical display names (if user passes arrays instead of column names)
    var_names = []
    for i, v in enumerate(xs):
        var_names.append(v if isinstance(v, str) else f"x{i}")

    def _get_arr(df, xcol, label):
        if df is None:
            return np.asarray([], float)

        # allow passing raw arrays
        if not isinstance(xcol, str):
            arr = to_array(xcol).astype(float, copy=False)
            return arr[np.isfinite(arr)]

        used = (
            _prep_covs(
                df, covariates,
                require_covariates=require_covariates,
                label=label
            )
            if residualize and covariates else []
        )

        if residualize and used:
            arr = residualize_preserve_mean(df, xcol, used).to_numpy(float)
        else:
            arr = pd.to_numeric(df[xcol], errors="coerce").to_numpy(float)

        return arr[np.isfinite(arr)]

    # -------- build long-form df --------

    rows = []
    for (sname, df, _c) in sample_defs:
        for j, canon_var in enumerate(var_names):
            sample_var = x_by_sample[sname][j]
            arr = _get_arr(df, sample_var, f"{sname} data")
            if arr.size:
                rows.append(pd.DataFrame({
                    "value": arr,
                    "variable": [canon_var] * arr.size,
                    "sample": [sname] * arr.size
                }))

    if not rows:
        ax.set_axis_off()
        return ax

    plot_df = pd.concat(rows, ignore_index=True)

    # -------- ordering --------

    var_order = list(order) if order is not None else list(var_names)
    if sample_order is None:
        sample_order = [s for (s, _, _) in sample_defs]
    else:
        sample_order = list(sample_order)

    palette = {s: c for (s, _, c) in sample_defs}

    # -------- barplot --------

    sns.barplot(
        data=plot_df,
        x="variable",
        y="value",
        hue="sample",
        order=var_order,
        hue_order=sample_order,
        estimator=estimator,
        errorbar=errorbar,
        capsize=capsize,
        palette=palette,
        saturation=saturation,
        ax=ax,
        **kwargs
    )

    # style bars
    for p in ax.patches:
        if hasattr(p, "set_alpha"):
            p.set_alpha(bar_alpha)
        if bar_edgecolor is not None and hasattr(p, "set_edgecolor"):
            p.set_edgecolor(bar_edgecolor)

    # -------- stripplot overlay --------

    if strip:
        sns.stripplot(
            data=plot_df,
            x="variable",
            y="value",
            hue="sample",
            order=var_order,
            hue_order=sample_order,
            dodge=True,
            jitter=strip_jitter,
            size=strip_size,
            alpha=strip_alpha,
            linewidth=strip_linewidth,
            palette=palette,
            ax=ax
        )

    # -------- legend (dedupe) --------

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        keep_h, keep_l, seen = [], [], set()
        for h, l in zip(handles, labels):
            if l in sample_order and l not in seen:
                seen.add(l)
                keep_h.append(h)
                keep_l.append(l)
        if keep_h:
            ax.legend(keep_h, keep_l, loc=legend_loc, frameon=False, fontsize=legend_fontsize)
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # -------- precompute arrays for p-values --------

    arr_lookup = {}
    for (sname, df, _c) in sample_defs:
        for j, canon_var in enumerate(var_names):
            sample_var = x_by_sample[sname][j]
            arr_lookup[(canon_var, sname)] = _get_arr(df, sample_var, f"{sname} data")

    def _p_to_stars(p):
        fn = globals().get("p_to_stars", None)
        if callable(fn):
            return fn(p)
        if not np.isfinite(p):
            return ""
        if p < 1e-3:
            return "***"
        if p < 1e-2:
            return "**"
        if p < 5e-2:
            return "*"
        return "n.s."

    # -------- robust bar->(variable,sample) assignment + annotation --------

    if annotate:
        # capture current limits BEFORE adding text
        y0, y1 = ax.get_ylim()
        yr = (y1 - y0) if (np.isfinite(y1 - y0) and y1 != y0) else 1.0

        # top anchor in data coordinates (uniform across all bars)
        # and per-sample stacked offsets (uniform across all variables)
        pad = pval_y_pad_frac * yr
        y_top_anchor = y0 + pval_top_frac * yr
        dy = pval_top_dy_frac * yr

        # Ensure headroom exists for the top stack
        if pval_top:
            # total stack height uses (n_hue-1)*dy plus a little pad
            needed = y_top_anchor + (len(sample_order) - 1) * dy + pad
            if needed > y1:
                ax.set_ylim(y0, needed + pad)
                y0, y1 = ax.get_ylim()
                yr = (y1 - y0) if y1 > y0 else 1.0
                y_top_anchor = y0 + pval_top_frac * yr
                dy = pval_top_dy_frac * yr

        xticks = np.asarray(ax.get_xticks(), float)
        xlabels = [t.get_text() for t in ax.get_xticklabels()]
        if not any(xlabels):
            xlabels = list(var_order)

        n_hue = len(sample_order)

        # Collect bar patches
        bar_patches = []
        for p in ax.patches:
            if not (hasattr(p, "get_x") and hasattr(p, "get_width") and hasattr(p, "get_height")):
                continue
            w = float(p.get_width())
            h = float(p.get_height())
            if not np.isfinite(w) or not np.isfinite(h) or w <= 0:
                continue
            bar_patches.append(p)

        # Group patches by nearest xtick (variable)
        groups = {i: [] for i in range(len(xticks))}
        for p in bar_patches:
            xc = float(p.get_x() + p.get_width() / 2.0)
            i_var = int(np.argmin(np.abs(xc - xticks)))
            if 0 <= i_var < len(xticks):
                groups[i_var].append(p)

        for i_var, patches in groups.items():
            if not patches:
                continue

            tick_x = float(xticks[i_var])
            var = xlabels[i_var] if i_var < len(xlabels) else None
            if var is None:
                continue

            # sort patches left->right within group
            patches = sorted(patches, key=lambda p: float(p.get_x() + p.get_width() / 2.0))

            # infer expected dodged offsets from bar width
            widths = np.array([float(p.get_width()) for p in patches], float)
            w = float(np.nanmedian(widths)) if np.isfinite(np.nanmedian(widths)) else float(patches[0].get_width())

            expected_offsets = (np.arange(n_hue) - (n_hue - 1) / 2.0) * w
            observed_offsets = np.array(
                [float(p.get_x() + p.get_width() / 2.0) - tick_x for p in patches],
                float
            )

            # assign observed bars to hue indices (handles missing bars)
            m = len(patches)
            hue_indices = range(n_hue)

            best_cost = np.inf
            best_assign = None
            for subset in itertools.combinations(hue_indices, m):
                for perm in itertools.permutations(subset):
                    cost = float(np.sum(np.abs(observed_offsets - expected_offsets[list(perm)])))
                    if cost < best_cost:
                        best_cost = cost
                        best_assign = list(perm)

            if best_assign is None:
                continue

            if debug:
                mapped = [(var, sample_order[h], float(observed_offsets[k])) for k, h in enumerate(best_assign)]
                print(f"[barstrip_samples] var={var} mapping={mapped} cost={best_cost:.4g}")

            # annotate each patch
            for p, hue_idx in zip(patches, best_assign):
                sample = sample_order[hue_idx]

                arr = arr_lookup.get((var, sample), None)
                if arr is None or arr.size == 0:
                    continue

                pval = pval_1samp(arr, popmean=popmean, tail=tail)
                star = _p_to_stars(pval)

                if (not pval_show_ns) and (star == "n.s."):
                    continue
                if star == "":
                    continue

                xc = float(p.get_x() + p.get_width() / 2.0)
                color = palette[sample] if (pval_color is None) else pval_color

                if pval_top:
                    # Uniform y-position: stack by sample (same y across all variables)
                    ytxt = y_top_anchor - hue_idx * dy
                    va = "center"
                else:
                    # Original behavior: at bar tip
                    y_a = float(p.get_y())
                    y_b = float(p.get_y() + p.get_height())
                    tip = y_b if abs(y_b) >= abs(y_a) else y_a
                    ytxt = tip + (pad if tip >= 0 else -pad)
                    va = "bottom" if tip >= 0 else "top"

                ax.text(
                    xc, ytxt, star,
                    ha="center", va=va,
                    fontsize=pval_fontsize,
                    color=color,
                    zorder=10,
                    clip_on=False
                )

    # -------- cosmetics --------
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    if despine:
        sns.despine(ax=ax)

    return ax

# cud v hc plots
def pval_group_effect_on_x(df, *, x, group_col, g0, g1, covariates=(), tail="two-sided"):
    """
    Test group effect on x (g1 - g0), optionally adjusting for covariates:
        x ~ 1 + I(group==g1) + covariates

    tail is applied to the group coefficient (g1 - g0).
    """
    covariates = _as_list(covariates)
    if df is None or df.empty:
        return np.nan

    df = df[df[group_col].isin([g0, g1])]
    if df.empty:
        return np.nan

    d = _numeric_dropna(df, [x] + covariates)

    # params: intercept + group + covariates
    k_params = 1 + 1 + len(covariates)
    if d.shape[0] <= k_params:
        return np.nan

    Y = d[x].to_numpy(float)
    g = (df.loc[d.index, group_col] == g1).astype(float).to_numpy(float)

    Xmat = (
        np.column_stack([g] + [d[c].to_numpy(float) for c in covariates])
        if covariates
        else g.reshape(-1, 1)
    )
    X = sm.add_constant(Xmat, has_constant="add")
    m = sm.OLS(Y, X).fit()

    # coefficient order: const, group, covariates...
    return tail_p_from_t(float(m.tvalues[1]), float(m.df_resid), tail)

def pval_beta_x_with_group(df, *, x, y, group_col, g0, g1, covariates=(), tail="two-sided"):
    """
    Test x slope while adjusting for group (and optional covariates):
        y ~ 1 + x + I(group==g1) + covariates

    tail is applied to beta_x.
    """
    covariates = _as_list(covariates)
    if df is None or df.empty:
        return np.nan

    df = df[df[group_col].isin([g0, g1])]
    if df.empty:
        return np.nan

    d = _numeric_dropna(df, [y, x] + covariates)

    # params: intercept + x + group + covariates
    k_params = 1 + 2 + len(covariates)
    if d.shape[0] <= k_params:
        return np.nan

    Y = d[y].to_numpy(float)
    g = (df.loc[d.index, group_col] == g1).astype(float).to_numpy(float)

    Xmat = np.column_stack(
        [d[x].to_numpy(float), g] + [d[c].to_numpy(float) for c in covariates]
    )
    X = sm.add_constant(Xmat, has_constant="add")
    m = sm.OLS(Y, X).fit()

    # coefficient order: const, x, group, covariates...
    return tail_p_from_t(float(m.tvalues[1]), float(m.df_resid), tail)

def histplot_groups(
    data=None,
    *,
    x,
    covariates=None,
    residualize=True,
    require_covariates=False,
    group_col="dx",
    groups=(0, 1),
    group_colors=None,          # if None: uses ctq_colors when group_col=="ctq", else dx_colors
    tail="two-sided",           # test direction for (g1 - g0)
    ax=None,
    bins=30,
    stat="density",
    kde=True,
    alpha=0.25,
    edgecolor=None,
    line_kws=None,
    vline0=False,
    mean_lines=True,
    mean_linestyle="--",
    mean_lw=1.2,
    mean_alpha=0.95,
    annotate=True,
    pval_x=0.5,
    pval_y=0.93,
    pval_fontsize=20,
    legend=True,
    legend_title="Group",
    set_xlim=True,
    pad_frac=0.05,
    return_xlim=False,
    **kwargs
):
    if ax is None:
        ax = plt.gca()
    if data is None:
        raise ValueError("histplot_groups expects `data` as a DataFrame.")
    if line_kws is None:
        line_kws = dict(linewidth=2.5, alpha=0.95)

    covariates = _as_list(covariates)
    g0, g1 = groups[0], groups[1]

    # --- colors: ctq -> ctq_colors, else -> dx_colors (unless explicitly provided)
    if group_colors is None:
        base = ctq_colors if str(group_col).lower() == "ctq" else dx_colors
        c0 = base[0] if len(base) >= 1 else "C0"
        c1 = base[1] if len(base) >= 2 else "C1"
    elif isinstance(group_colors, dict):
        c0, c1 = group_colors.get(g0, "C0"), group_colors.get(g1, "C1")
    elif isinstance(group_colors, (list, tuple)) and len(group_colors) >= 2:
        c0, c1 = group_colors[0], group_colors[1]
    else:
        c0, c1 = "C0", "C1"

    # --- labels (optional globals; fall back to raw values)
    _labels = globals().get(f"{group_col}_labels", None)
    if not isinstance(_labels, dict):
        # common fallback for dx if user defined dx_labels
        if str(group_col).lower() == "dx":
            _labels = globals().get("dx_labels", None)
        _labels = _labels if isinstance(_labels, dict) else None
    lab0 = _labels.get(g0, str(g0)) if isinstance(_labels, dict) else str(g0)
    lab1 = _labels.get(g1, str(g1)) if isinstance(_labels, dict) else str(g1)

    df_use = data[data[group_col].isin([g0, g1])].copy()

    # --- build plotted x vector (residualized or raw)
    if isinstance(x, str):
        used_covs = _prep_covs(df_use, covariates, require_covariates=require_covariates, label="Data")
        if residualize and used_covs:
            df_use["_x_plot"] = residualize_preserve_mean(df_use, x, used_covs)
            xlabel = f"{x} (adj.)"
        else:
            df_use["_x_plot"] = pd.to_numeric(df_use[x], errors="coerce")
            xlabel = x
        x_for_test = x
        can_ancova = True
    else:
        # allow Series aligned to data, or arrays matching len(data) or len(df_use)
        if isinstance(x, pd.Series):
            s = pd.to_numeric(x, errors="coerce")
            if s.index.equals(data.index):
                df_use["_x_plot"] = s.loc[df_use.index]
            else:
                df_use["_x_plot"] = s.reindex(df_use.index)
        else:
            arr = np.asarray(x, float)
            if arr.size == len(data):
                df_use["_x_plot"] = pd.Series(arr, index=data.index).loc[df_use.index]
            elif arr.size == len(df_use):
                df_use["_x_plot"] = pd.Series(arr, index=df_use.index)
            else:
                raise ValueError("Array-like x must match len(data) or len(filtered df).")
        xlabel = "x"
        used_covs = []
        x_for_test = "_x_plot"
        can_ancova = False

    d0 = df_use[df_use[group_col] == g0]
    d1 = df_use[df_use[group_col] == g1]
    x0 = to_array("_x_plot", d0)
    x1 = to_array("_x_plot", d1)

    # --- xlim
    xlim = None
    if set_xlim:
        xlim = auto_xlim([x0, x1], pad_frac=pad_frac)

    # --- plot
    if x0.size:
        sns.histplot(
            x=x0, bins=bins, stat=stat, kde=kde,
            color=c0, alpha=alpha, edgecolor=edgecolor,
            line_kws=line_kws, ax=ax, label=lab0, **kwargs
        )
    if x1.size:
        sns.histplot(
            x=x1, bins=bins, stat=stat, kde=kde,
            color=c1, alpha=alpha, edgecolor=edgecolor,
            line_kws=line_kws, ax=ax, label=lab1, **kwargs
        )

    if mean_lines and x0.size:
        ax.axvline(np.mean(x0), color=c0, linestyle=mean_linestyle, linewidth=mean_lw, alpha=mean_alpha)
    if mean_lines and x1.size:
        ax.axvline(np.mean(x1), color=c1, linestyle=mean_linestyle, linewidth=mean_lw, alpha=mean_alpha)
    if vline0:
        ax.axvline(0, color="black", linestyle="-", linewidth=1)

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(stat)

    # --- stats + stars
    if annotate:
        if can_ancova and isinstance(x_for_test, str):
            p = pval_group_effect_on_x(
                df_use, x=x_for_test, group_col=group_col, g0=g0, g1=g1,
                covariates=used_covs, tail=tail
            )
        else:
            p = pval_t_ind(x0, x1, tail=tail, equal_var=False)

        add_pval(p, ax, x=pval_x, y=pval_y, fontsize=pval_fontsize, color="black")

    if legend:
        ax.legend(title=legend_title)

    if return_xlim:
        return ax, (None, None) if xlim is None else xlim
    return ax

def regplot_groups(
    data=None,
    *,
    x,
    y,
    covariates=None,
    residualize=True,
    tail="greater",
    require_covariates=False,
    group_col="dx",
    groups=(0, 1),
    group_colors=None,          # if None: uses ctq_colors when group_col=="ctq", else dx_colors
    ax=None,
    set_xlim=None,              # None | (xmin,xmax) | "auto"
    pad_frac=0.05,
    annotate=True,
    pval_x=0.5,
    pval_y=0.93,
    pval_fontsize=20,
    scatter=True,
    scatter_kws=None,
    line_kws=None,
    vline0=False,
    lines="separate",           # "separate" | "combined"
    combined_color="black",
):
    if ax is None:
        ax = plt.gca()
    if data is None:
        raise ValueError("regplot_groups expects `data` as a DataFrame.")

    covariates = _as_list(covariates)
    g0, g1 = groups[0], groups[1]

    if scatter_kws is None:
        scatter_kws = dict(alpha=0.25, s=18)
    if line_kws is None:
        line_kws = dict(alpha=0.95, linewidth=2.5)

    # --- colors: ctq -> ctq_colors, else -> dx_colors (unless explicitly provided)
    if group_colors is None:
        base = ctq_colors if str(group_col).lower() == "ctq" else dx_colors
        c0 = base[0] if len(base) >= 1 else "C0"
        c1 = base[1] if len(base) >= 2 else "C1"
    elif isinstance(group_colors, dict):
        c0, c1 = group_colors.get(g0, "C0"), group_colors.get(g1, "C1")
    elif isinstance(group_colors, (list, tuple)) and len(group_colors) >= 2:
        c0, c1 = group_colors[0], group_colors[1]
    else:
        c0, c1 = "C0", "C1"

    # --- labels (optional globals; fall back to raw values)
    _labels = globals().get(f"{group_col}_labels", None)
    if not isinstance(_labels, dict):
        if str(group_col).lower() == "dx":
            _labels = globals().get("dx_labels", None)
        _labels = _labels if isinstance(_labels, dict) else None
    lab0 = _labels.get(g0, str(g0)) if isinstance(_labels, dict) else str(g0)
    lab1 = _labels.get(g1, str(g1)) if isinstance(_labels, dict) else str(g1)

    df_all = data[data[group_col].isin([g0, g1])].copy()
    used_covs = _prep_covs(df_all, covariates, require_covariates=require_covariates, label="Data")

    # --- plotted relationship (raw vs residualized)
    do_resid = residualize and bool(used_covs)
    if do_resid:
        df_plot = df_all.copy()
        df_plot["_x_plot"] = residualize_preserve_mean_by(df_plot, x, used_covs, by=group_col, groups=[g0, g1])
        df_plot["_y_plot"] = residualize_preserve_mean_by(df_plot, y, used_covs, by=group_col, groups=[g0, g1])
        x_plot, y_plot = "_x_plot", "_y_plot"
        xlab, ylab = f"{x} (adj.)", f"{y} (adj.)"
    else:
        df_plot = df_all
        x_plot, y_plot = x, y
        xlab, ylab = x, y

    # --- xlim
    if set_xlim == "auto":
        xlim = auto_xlim([np.asarray(df_plot[x_plot], float)], pad_frac=pad_frac)
        set_xlim = xlim if xlim is not None else None

    # --- plot
    d0 = df_plot[df_plot[group_col] == g0]
    d1 = df_plot[df_plot[group_col] == g1]

    if scatter:
        sns.scatterplot(data=d0, x=x_plot, y=y_plot, color=c0, label=lab0, ax=ax, **scatter_kws)
        sns.scatterplot(data=d1, x=x_plot, y=y_plot, color=c1, label=lab1, ax=ax, **scatter_kws)

    if lines == "separate":
        sns.regplot(data=d0, x=x_plot, y=y_plot, scatter=False, color=c0, line_kws=line_kws, ax=ax)
        sns.regplot(data=d1, x=x_plot, y=y_plot, scatter=False, color=c1, line_kws=line_kws, ax=ax)
    elif lines == "combined":
        sns.regplot(data=df_plot, x=x_plot, y=y_plot, scatter=False, color=combined_color, line_kws=line_kws, ax=ax)
    else:
        raise ValueError("lines must be 'separate' or 'combined'.")

    if vline0:
        ax.axvline(0, color="black", linestyle="-", linewidth=1)

    if isinstance(set_xlim, (tuple, list)) and len(set_xlim) == 2:
        ax.set_xlim(float(set_xlim[0]), float(set_xlim[1]))

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # --- stats + stars
    if annotate:
        if lines == "separate":
            p0 = pval_beta_x(df_all[df_all[group_col] == g0], x=x, y=y, covariates=used_covs, tail=tail)
            p1 = pval_beta_x(df_all[df_all[group_col] == g1], x=x, y=y, covariates=used_covs, tail=tail)

            add_pval(p0, ax, x=pval_x, y=pval_y, fontsize=pval_fontsize, color=c0)
            add_pval(p1, ax, x=pval_x, y=pval_y - 0.08, fontsize=pval_fontsize, color=c1)
        else:
            p = pval_beta_x_with_group(
                df_all, x=x, y=y, group_col=group_col, g0=g0, g1=g1,
                covariates=used_covs, tail=tail
            )
            add_pval(p, ax, x=pval_x, y=pval_y, fontsize=pval_fontsize, color=combined_color)

    ax.legend(title="Group", frameon=False, fontsize=8, handlelength=1.6, borderaxespad=0.2, labelspacing=0.2)
    return ax

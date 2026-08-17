from utils import *
from utils_fmri import regress_out_temporal_trend

import scipy.stats
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.base import clone

results_dir = "../results/roi_analysis/"
os.makedirs(results_dir, exist_ok=True)

out_csv = os.path.join(results_dir, "dimension-decoding.csv")
if os.path.exists(out_csv):
    results_existing = pd.read_csv(out_csv)
    processed = set(zip(results_existing["sub_id"], results_existing["roi"]))
else:
    results_existing = None
    processed = set()

subject_data = load_pickle("../analyses/lsa_decision_spm/subject_data_tavares-striatum.pkl")

# labels / nuisance design
character_id = decision_trials["character_role_num"].to_numpy()
chars = pd.get_dummies(
    pd.Series(character_id, name="char"),
    drop_first=True
).values
X_char = np.column_stack([np.ones(len(chars)), chars])

dimension_labels = decision_trials["dimension"].to_numpy()
labels_affil     = (dimension_labels == "affil").astype(int)
onsets = decision_trials["onset"].to_numpy()

# Global trial count (ensures consistent columns across all rows)
T = len(labels_affil)
pred_col_width = max(2, len(str(T)))
pred_cols = [f"decision_{i+1:0{pred_col_width}d}" for i in range(T)]

# helpers
def circular_shift_null_accuracy(
    estimator,
    X,
    y,
    outer_cv,
    n_perm=100,
    seed=0,
):
    rng = np.random.default_rng(seed)
    Tloc = len(y)
    shifts = rng.integers(1, Tloc, size=n_perm)

    null_accs = np.empty(n_perm, dtype=float)
    for i, s in enumerate(shifts):
        y_perm = np.roll(y, s)
        null_accs[i] = cross_val_score(
            estimator,
            X,
            y_perm,
            cv=outer_cv,
            scoring="accuracy",
            n_jobs=1,   # avoid oversubscription since GridSearchCV uses n_jobs=-1
        ).mean()
    return null_accs

def nested_cv_oof_predict_proba_and_accuracy(search_estimator, X, y, outer_cv):
    """
    Returns:
      oof_proba: (n_trials,) out-of-fold predicted probabilities for class 1
      mean_acc:  mean accuracy across outer folds (matches your previous approach)
    """
    oof_proba = np.full(len(y), np.nan, dtype=float)
    fold_accs = []

    for train_idx, test_idx in outer_cv.split(X, y):
        est = clone(search_estimator)
        est.fit(X[train_idx], y[train_idx])

        # Probability of class "1" (affil)
        p = est.predict_proba(X[test_idx])[:, 1]
        oof_proba[test_idx] = p

        # Fold accuracy on held-out data
        y_hat = est.predict(X[test_idx])
        fold_accs.append(accuracy_score(y[test_idx], y_hat))

    return oof_proba, float(np.mean(fold_accs))

# classifier setup
base_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000,
    )),
])

n_perm = 50
alphas = np.logspace(-4, 4, 3)
Cs = 1.0 / alphas
param_grid = {"clf__C": Cs}

outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
search   = GridSearchCV(
    estimator=base_pipe,
    param_grid=param_grid,
    scoring="accuracy",
    cv=inner_cv,
    n_jobs=-1,
)

rois = ['HPC-L', 'HPC-R', 'PCC-L', 'PCC-R', 'DLPFC-L', 'DLPFC-R']

# fixed column order for writing
base_cols = ["sub_id", "roi", "obs", "null_mean", "null_sd", "z"]
all_cols = base_cols + pred_cols
for roi in tqdm(rois, desc="ROIs"):
    for sub_id in tqdm(incl_subs, desc=f"{roi}", leave=False):

        # skip if already computed
        if (sub_id, roi) in processed:
            continue

        # Initialize row with NaNs including ALL prediction columns
        row = {c: np.nan for c in all_cols}
        row["sub_id"] = sub_id
        row["roi"] = roi

        try:
            betas = subject_data[sub_id]["roi_betas"][roi]
            n_trials, n_voxels = betas.shape

            # Basic safety check: ensure trial alignment
            if n_trials != T:
                raise ValueError(f"Trial count mismatch: betas has {n_trials}, labels have {T}")

            # ----------------------------------------------------------
            # regress out character identity
            # ----------------------------------------------------------

            coef = np.linalg.lstsq(X_char, betas, rcond=None)[0]
            betas_res = betas - X_char @ coef

            # ----------------------------------------------------------
            # remove temporal autocorrelation
            # ----------------------------------------------------------

            X = regress_out_temporal_trend(betas_res, onsets)

            # ----------------------------------------------------------
            # observed accuracy + out-of-fold trial-wise probabilities
            # ----------------------------------------------------------

            oof_proba, obs_acc = nested_cv_oof_predict_proba_and_accuracy(
                search, X, labels_affil, outer_cv
            )

            row.update(dict(zip(pred_cols, oof_proba)))

            # ----------------------------------------------------------
            # permutation null
            # ----------------------------------------------------------
            
            null_accs = circular_shift_null_accuracy(
                search,
                X,
                labels_affil,
                outer_cv,
                n_perm=n_perm,
                seed=0,
            )

            null_mean = null_accs.mean()
            null_sd = null_accs.std(ddof=1)
            z_acc = (obs_acc - null_mean) / null_sd if null_sd > 0 else np.nan

            row.update({
                "obs": obs_acc,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "z": z_acc,
            })

        except Exception:
            pass

        pd.DataFrame([row], columns=all_cols).to_csv(
            out_csv,
            mode="a",
            header=not os.path.exists(out_csv),
            index=False,
        )

        processed.add((sub_id, roi))

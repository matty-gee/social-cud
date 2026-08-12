"""Composable prototype learner models and simulation utilities.

This module keeps the original prototype/episodic behavior while adding a
component-based model builder so new variants can be assembled from the same
retrieval, update, storage, and learning-rate pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import scipy.stats
from tqdm.auto import tqdm

# ============================================================
# Type aliases
# ============================================================

State = dict[str, Any]
Vec = np.ndarray
Info = dict[str, Any]

RetrieveFn = Callable[[State, int], tuple[Vec, Info]]
StoreFn = Callable[[State, int, Vec, Mapping[str, Any], Mapping[str, Any]], State]
LearningRateFn = Callable[..., float]
UpdateFn = Callable[[Vec, Vec, float], tuple[Vec, Info]]
WeightsFn = Callable[[State, int], tuple[np.ndarray | None, np.ndarray | None]]
InitFn = Callable[[int], State]


RETRIEVAL_FIELD_DEFAULTS = dict(
    retrieved_affil=np.nan,
    retrieved_power=np.nan,
    mem_n=0,
    retrieval_entropy=np.nan,
    retrieval_neff=np.nan,
    other_mass=np.nan,
)


# ============================================================
# Data helpers
# ============================================================

def encode_characters(char_series: pd.Series):
    """Map character IDs to contiguous 0..C-1 codes for fast array-based models."""
    uniq = pd.unique(char_series)
    char_to_idx = {c: i for i, c in enumerate(uniq)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    codes = char_series.map(char_to_idx).to_numpy()
    return codes, char_to_idx, idx_to_char


def make_episode_df(
    choices: np.ndarray,
    character_role_num: np.ndarray,
    *,
    normalize: bool = True,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Build a canonical long-format episode DataFrame from trialwise choices."""
    x = np.asarray(choices, dtype=float)
    chars = np.asarray(character_role_num)

    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"`choices` must be shape (T,2); got {x.shape}")
    t = x.shape[0]

    if chars.ndim != 1:
        chars = chars.reshape(-1)
    if chars.shape[0] != t:
        raise ValueError(
            f"`character_role_num` must have length T={t}; got len={chars.shape[0]}"
        )

    df = pd.DataFrame(
        {
            "global_t": np.arange(1, t + 1, dtype=int),
            "character_role_num": chars,
            "choice_affil": x[:, 0],
            "choice_power": x[:, 1],
        }
    )

    df["decision_num"] = df.groupby("character_role_num").cumcount() + 1

    if normalize:
        norms = np.linalg.norm(x, axis=1)
        valid = norms > eps
        df["choice_affil_unit"] = np.nan
        df["choice_power_unit"] = np.nan
        df.loc[valid, "choice_affil_unit"] = x[valid, 0] / norms[valid]
        df.loc[valid, "choice_power_unit"] = x[valid, 1] / norms[valid]

    df["pos_affil"] = df.groupby("character_role_num")["choice_affil"].cumsum()
    df["pos_power"] = df.groupby("character_role_num")["choice_power"].cumsum()
    return df


# ============================================================
# Core state + learning
# ============================================================

def init_proto_state(n_char: int) -> State:
    return {
        "n_char": int(n_char),
        "mu": np.zeros((int(n_char), 2), float),
        "n": np.zeros(int(n_char), int),
        "mem_vecs": [],
        "mem_chars": [],
    }


def _vec2(x: Any) -> np.ndarray:
    return np.asarray(x, float).reshape(2,)


def lr_running_mean(state: State, char_idx: int, *, alpha: float | None) -> float:
    """alpha=None -> exact running mean (1/n_c), alpha=float -> leaky mean."""
    if alpha is None:
        state["n"][char_idx] += 1
        return 1.0 / state["n"][char_idx]

    lr = float(alpha)
    if not (0.0 < lr <= 1.0):
        raise ValueError(f"alpha must be in (0,1]; got {lr}")
    state["n"][char_idx] += 1
    return lr


def delta_rule_update(base: np.ndarray, x: np.ndarray, lr: float) -> tuple[np.ndarray, Info]:
    """m_new = base + lr*(x-base)."""
    base = _vec2(base)
    x = _vec2(x)

    delta = x - base
    update = lr * delta
    m_new = base + update

    info = {
        "affil_delta": float(delta[0]),
        "power_delta": float(delta[1]),
        "affil_update": float(update[0]),
        "power_update": float(update[1]),
        "update_magnitude": float(np.linalg.norm(update)),
        "surprise": float(delta @ delta),
    }
    return m_new, info


# ============================================================
# Retrieval
# ============================================================

def retrieve_none(state: State, char_idx: int, **_kwargs) -> tuple[np.ndarray, Info]:
    """No episodic retrieval: base is current prototype."""
    m_prev = state["mu"][char_idx]
    info = {
        "retrieved_affil": float(m_prev[0]),
        "retrieved_power": float(m_prev[1]),
        "mem_n": int(len(state.get("mem_vecs", []))),
        "retrieval_entropy": np.nan,
        "retrieval_neff": np.nan,
        "other_mass": np.nan,
    }
    return m_prev, info


def _safe_softmax(scores: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    scores = np.asarray(scores, float)
    scores = scores - np.max(scores)
    w = np.exp(scores)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return np.full_like(w, 1.0 / len(w))
    return w / (s + eps)


def retrieve_softmax_episode(
    state: State,
    char_idx: int,
    *,
    kappa: float,
    same_char_bias: float = 0.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, Info]:
    """Query q = current mu[char_idx]. Retrieve convex combo of stored episodes."""
    m_prev = state["mu"][char_idx]
    mem_vecs = state.get("mem_vecs", [])
    mem_chars = state.get("mem_chars", [])

    if len(mem_vecs) == 0:
        info = {
            "retrieved_affil": float(m_prev[0]),
            "retrieved_power": float(m_prev[1]),
            "mem_n": 0,
            "retrieval_entropy": 0.0,
            "retrieval_neff": 0.0,
            "other_mass": 0.0,
        }
        return m_prev, info

    e = np.asarray(mem_vecs, float)  # (M, 2)
    c = np.asarray(mem_chars, int)  # (M,)

    q = m_prev
    sims = e @ q

    bias = (c == char_idx).astype(float) * float(same_char_bias)
    w = _safe_softmax(float(kappa) * sims + bias, eps=eps)

    retrieved = (w[:, None] * e).sum(axis=0)

    entropy = float(-np.sum(w * np.log(w + eps)))
    neff = float(1.0 / np.sum(w * w + eps))
    other_mass = float(w[c != char_idx].sum())

    info = {
        "retrieved_affil": float(retrieved[0]),
        "retrieved_power": float(retrieved[1]),
        "mem_n": int(len(mem_vecs)),
        "retrieval_entropy": entropy,
        "retrieval_neff": neff,
        "other_mass": other_mass,
    }
    return retrieved, info


# ============================================================
# Storage
# ============================================================

def store_mu_only(
    state: State,
    char_idx: int,
    m_new: np.ndarray,
    *_args,
    **_kwargs,
) -> State:
    state["mu"][char_idx] = _vec2(m_new)
    return state


def store_mu_and_episode(
    state: State,
    char_idx: int,
    m_new: np.ndarray,
    *_args,
    **_kwargs,
) -> State:
    m_new = _vec2(m_new)
    state["mu"][char_idx] = m_new
    state.setdefault("mem_vecs", []).append(m_new.copy())
    state.setdefault("mem_chars", []).append(int(char_idx))
    return state


# ============================================================
# Composable components
# ============================================================

@dataclass(frozen=True)
class ModelComponents:
    """Composable model primitives for online updates."""

    retrieve_fn: RetrieveFn
    store_fn: StoreFn
    lr_fn: LearningRateFn = lr_running_mean
    update_fn: UpdateFn = delta_rule_update


def make_components(
    *,
    retrieve_fn: RetrieveFn,
    store_fn: StoreFn,
    lr_fn: LearningRateFn = lr_running_mean,
    update_fn: UpdateFn = delta_rule_update,
) -> ModelComponents:
    return ModelComponents(
        retrieve_fn=retrieve_fn,
        store_fn=store_fn,
        lr_fn=lr_fn,
        update_fn=update_fn,
    )


def _resolve_components(
    *,
    components: ModelComponents | None,
    retrieve_fn: RetrieveFn | None,
    store_fn: StoreFn | None,
    lr_fn: LearningRateFn | None,
    update_fn: UpdateFn | None,
) -> ModelComponents:
    if components is not None:
        if any(fn is not None for fn in (retrieve_fn, store_fn, lr_fn, update_fn)):
            raise ValueError("Pass either `components` or individual functions, not both.")
        return components

    if retrieve_fn is None or store_fn is None:
        raise ValueError(
            "Missing required functions: provide `components` or both "
            "`retrieve_fn` and `store_fn`."
        )

    return make_components(
        retrieve_fn=retrieve_fn,
        store_fn=store_fn,
        lr_fn=lr_fn or lr_running_mean,
        update_fn=update_fn or delta_rule_update,
    )


# ============================================================
# Output normalization helper
# ============================================================

def fill_retrieval_fields(out: Info) -> Info:
    """Ensure retrieval fields exist (prototype models will have NaNs)."""
    for k, v in RETRIEVAL_FIELD_DEFAULTS.items():
        out.setdefault(k, v)
    return out


# ============================================================
# Online step + runner
# ============================================================

def online_step(
    *,
    state: State,
    char_idx: int,
    x: np.ndarray,
    components: ModelComponents | None = None,
    retrieve_fn: RetrieveFn | None = None,
    lr_fn: LearningRateFn | None = None,
    update_fn: UpdateFn | None = None,
    store_fn: StoreFn | None = None,
    alpha: float | None = None,
) -> tuple[State, Info]:
    """One online step. Returns (new_state, out_dict)."""
    components = _resolve_components(
        components=components,
        retrieve_fn=retrieve_fn,
        store_fn=store_fn,
        lr_fn=lr_fn,
        update_fn=update_fn,
    )
    x = _vec2(x)

    if not np.all(np.isfinite(x)):
        m = state["mu"][char_idx]
        out = {
            "affil_mean": float(m[0]),
            "power_mean": float(m[1]),
            "affil_delta": np.nan,
            "power_delta": np.nan,
            "affil_update": 0.0,
            "power_update": 0.0,
            "mean_magnitude": float(np.linalg.norm(m)),
            "update_magnitude": 0.0,
            "surprise": np.nan,
            "alpha": np.nan,
            "retrieved_affil": float(m[0]),
            "retrieved_power": float(m[1]),
            "mem_n": int(len(state.get("mem_vecs", []))),
            "retrieval_entropy": np.nan,
            "retrieval_neff": np.nan,
            "other_mass": np.nan,
        }
        return state, out

    base, rinfo = components.retrieve_fn(state, char_idx)
    lr = components.lr_fn(state, char_idx, alpha=alpha)
    m_new, uinfo = components.update_fn(base, x, lr)
    state = components.store_fn(state, char_idx, m_new, rinfo, uinfo)

    out = {
        "affil_mean": float(m_new[0]),
        "power_mean": float(m_new[1]),
        "mean_magnitude": float(np.linalg.norm(m_new)),
        "alpha": float(lr),
        **uinfo,
        **rinfo,
    }
    return state, out


def run_online_model_functional(
    episodes: pd.DataFrame,
    *,
    components: ModelComponents | None = None,
    retrieve_fn: RetrieveFn | None = None,
    store_fn: StoreFn | None = None,
    lr_fn: LearningRateFn | None = None,
    update_fn: UpdateFn | None = None,
    alpha: float | None = None,
    x_cols: tuple[str, str] = ("choice_affil_unit", "choice_power_unit"),
) -> tuple[pd.DataFrame, State]:
    ep = episodes.sort_values("global_t").reset_index(drop=True)

    if not set(x_cols).issubset(ep.columns):
        raise ValueError(f"Episodes missing x_cols={x_cols}")

    components = _resolve_components(
        components=components,
        retrieve_fn=retrieve_fn,
        store_fn=store_fn,
        lr_fn=lr_fn,
        update_fn=update_fn,
    )

    char_codes, char_to_idx, idx_to_char = encode_characters(ep["character_role_num"])
    n_char = int(np.max(char_codes)) + 1
    state = init_proto_state(n_char)
    state["char_to_idx"] = dict(char_to_idx)
    state["idx_to_char"] = dict(idx_to_char)

    out_rows = []
    for i, row in enumerate(ep.itertuples(index=False)):
        x = np.array([getattr(row, x_cols[0]), getattr(row, x_cols[1])], float)
        state, out = online_step(
            state=state,
            char_idx=int(char_codes[i]),
            x=x,
            components=components,
            alpha=alpha,
        )
        out_rows.append(fill_retrieval_fields(out))

    out_df = pd.DataFrame(out_rows)
    return pd.concat([ep, out_df], axis=1), state


# ============================================================
# Readout helpers (state-based)
# ============================================================

def linreg_slope(x, y):
    """Return OLS slope of y ~ x (with intercept)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x = x[m]
    y = y[m]
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom <= 0:
        return np.nan
    return float((xc @ (y - y.mean())) / denom)


def pearson_r(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    return scipy.stats.pearsonr(np.asarray(x)[m], np.asarray(y)[m])[0]


def distance_matrix_from_mu(state: State) -> np.ndarray:
    """Pairwise Euclidean distances between learned prototypes mu_i."""
    mu = np.asarray(state["mu"], float)  # (n_char,2)
    diffs = mu[:, None, :] - mu[None, :, :]
    return np.linalg.norm(diffs, axis=2)


def retrieval_weights(
    state: State,
    *,
    cue_char: int,
    kappa: float,
    same_char_bias: float = 0.0,
    restrict: str = "all",  # "all" | "same"
    eps: float = 1e-12,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    mem_vecs = state.get("mem_vecs", [])
    mem_chars = state.get("mem_chars", [])
    if len(mem_vecs) == 0:
        return None, None

    e = np.asarray(mem_vecs, float)  # (M,2)
    c = np.asarray(mem_chars, int)  # (M,)
    q = _vec2(state["mu"][int(cue_char)])

    if restrict == "same":
        mask = c == int(cue_char)
    else:
        mask = np.ones(len(c), dtype=bool)

    if not np.any(mask):
        return None, None

    idx = np.flatnonzero(mask)
    e2 = e[idx]
    c2 = c[idx]

    sims = e2 @ q
    bias = (c2 == int(cue_char)).astype(float) * float(same_char_bias)
    scores = float(kappa) * sims + bias
    scores = scores - np.max(scores)
    w = np.exp(scores)
    w /= w.sum() + eps

    return w, idx


def predicted_confusability_mass(
    state: State,
    *,
    kappa: float,
    same_char_bias: float = 0.0,
    restrict: str = "all",
) -> np.ndarray:
    n = int(state["n_char"])
    m = np.zeros((n, n), dtype=float)
    mem_chars_all = np.asarray(state.get("mem_chars", []), dtype=int)

    for cue in range(n):
        w, idx = retrieval_weights(
            state,
            cue_char=cue,
            kappa=kappa,
            same_char_bias=same_char_bias,
            restrict=restrict,
        )
        if w is None or idx is None:
            continue

        c_sel = mem_chars_all[idx]
        for j in range(n):
            m[cue, j] = float(w[c_sel == j].sum())

    return m


def confusion_prob_offdiag(m):
    m = np.asarray(m, float)
    coff = m.copy()
    np.fill_diagonal(coff, 0.0)
    row_sums = coff.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cprob = coff / row_sums
    cprob[~np.isfinite(cprob)] = 0.0
    return cprob


def confusability_vs_distance_slope(
    state: State,
    *,
    kappa: float,
    same_char_bias: float = 0.0,
    restrict: str = "all",
):
    m = predicted_confusability_mass(
        state,
        kappa=kappa,
        same_char_bias=same_char_bias,
        restrict=restrict,
    )
    cprob = confusion_prob_offdiag(m)
    d = distance_matrix_from_mu(state)

    n = cprob.shape[0]
    xs, ys = [], []
    for i in range(n):
        for j in range(i + 1, n):
            xs.append(float(d[i, j]))
            ys.append(float(0.5 * (cprob[i, j] + cprob[j, i])))

    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    slope = linreg_slope(xs, ys)
    r = pearson_r(xs, ys)
    return slope, r


def rt_correlations(df, *, rt_col="reaction_time"):
    for c in [rt_col, "retrieval_entropy", "retrieval_neff"]:
        if c not in df.columns:
            return np.nan, np.nan
    return pearson_r(df[rt_col], df["retrieval_entropy"]), pearson_r(
        df[rt_col], df["retrieval_neff"]
    )


# ============================================================
# Model spec composition
# ============================================================

def _default_retrieve_state_fn(retrieve_fn: RetrieveFn):
    def retrieve_state_fn(state, cue_char: int, **_kwargs):
        v, rinfo = retrieve_fn(state, int(cue_char))
        info = dict(
            entropy=rinfo.get("retrieval_entropy", np.nan),
            n_eff=rinfo.get("retrieval_neff", np.nan),
            other_mass=rinfo.get("other_mass", np.nan),
        )
        return v, info

    return retrieve_state_fn


def build_model_spec(
    *,
    name: str,
    components: ModelComponents,
    alpha: float | None = None,
    init_fn: InitFn = init_proto_state,
    retrieve_state_fn: Callable[[State, int], tuple[Vec, Info]] | None = None,
    weights_fn: WeightsFn | None = None,
    readout_params: Mapping[str, Any] | None = None,
    rt_col: str = "reaction_time",
) -> dict[str, Any]:
    """Build a model spec dict from composable components."""
    if retrieve_state_fn is None:
        retrieve_state_fn = _default_retrieve_state_fn(components.retrieve_fn)
    if weights_fn is None:
        weights_fn = lambda *_args, **_kwargs: (None, None)

    def step_fn(state, char_idx: int, x: np.ndarray, **_kwargs):
        state, out = online_step(
            state=state,
            char_idx=int(char_idx),
            x=x,
            components=components,
            alpha=alpha,
        )
        return state, fill_retrieval_fields(out)

    spec = {
        "name": str(name),
        "init_fn": init_fn,
        "step_fn": step_fn,
        "retrieve_state_fn": retrieve_state_fn,
        "weights_fn": weights_fn,
        "components": components,
        "retrieve_fn": components.retrieve_fn,
        "store_fn": components.store_fn,
        "lr_fn": components.lr_fn,
        "update_fn": components.update_fn,
        "alpha": alpha,
        "rt_col": rt_col,
    }
    if readout_params:
        spec.update(dict(readout_params))
    return spec


def compose_model_spec(
    *,
    name: str,
    retrieve_fn: RetrieveFn,
    store_fn: StoreFn,
    lr_fn: LearningRateFn = lr_running_mean,
    update_fn: UpdateFn = delta_rule_update,
    alpha: float | None = None,
    init_fn: InitFn = init_proto_state,
    retrieve_state_fn: Callable[[State, int], tuple[Vec, Info]] | None = None,
    weights_fn: WeightsFn | None = None,
    readout_params: Mapping[str, Any] | None = None,
    rt_col: str = "reaction_time",
) -> dict[str, Any]:
    """Convenience wrapper to compose a model spec from primitive functions."""
    components = make_components(
        retrieve_fn=retrieve_fn,
        store_fn=store_fn,
        lr_fn=lr_fn,
        update_fn=update_fn,
    )
    return build_model_spec(
        name=name,
        components=components,
        alpha=alpha,
        init_fn=init_fn,
        retrieve_state_fn=retrieve_state_fn,
        weights_fn=weights_fn,
        readout_params=readout_params,
        rt_col=rt_col,
    )


def make_model_spec_episodic(*, alpha=None, kappa=8.0, same_char_bias=0.0, restrict="all"):
    kappa = float(kappa)
    same_char_bias = float(same_char_bias)

    retrieve_step_fn = partial(
        retrieve_softmax_episode,
        kappa=kappa,
        same_char_bias=same_char_bias,
        eps=1e-12,
    )

    def weights_fn(state, cue_char: int, **_kwargs):
        return retrieval_weights(
            state,
            cue_char=int(cue_char),
            kappa=kappa,
            same_char_bias=same_char_bias,
            restrict=str(restrict),
        )

    return compose_model_spec(
        name="episodic",
        retrieve_fn=retrieve_step_fn,
        store_fn=store_mu_and_episode,
        alpha=alpha,
        weights_fn=weights_fn,
        readout_params={
            "kappa": kappa,
            "same_char_bias": same_char_bias,
            "restrict": restrict,
        },
    )


def make_model_spec_prototype(*, alpha=None):
    def retrieve_state_fn(state, cue_char: int, **_kwargs):
        v, _ = retrieve_none(state, int(cue_char))
        info = dict(entropy=np.nan, n_eff=np.nan, other_mass=np.nan)
        return v, info

    return compose_model_spec(
        name="prototype",
        retrieve_fn=retrieve_none,
        store_fn=store_mu_only,
        alpha=alpha,
        retrieve_state_fn=retrieve_state_fn,
        readout_params={
            "kappa": None,
            "same_char_bias": None,
            "restrict": "all",
        },
    )


# ============================================================
# Fit + summarize subjects
# ============================================================

def fit_subject(
    sub_id,
    *,
    model_spec: Mapping[str, Any],
    neutrals=False,
    normalize=True,
):
    """Fit one subject using an externally provided `load_behavior` function."""
    if "load_behavior" not in globals():
        raise NameError("`load_behavior` is not defined in this module scope.")

    behavior = load_behavior(sub_id, neutrals=neutrals, on_missing="none")
    if behavior is None or len(behavior) == 0:
        return None, None, None

    char_col = (
        "character_role_num"
        if "character_role_num" in behavior.columns
        else "char_role_num"
    )
    choices = behavior[["affil_decision", "power_decision"]].to_numpy(float)
    chars = behavior[char_col].to_numpy()

    episodes = make_episode_df(choices, chars, normalize=normalize)

    df, state = run_online_model_functional(
        episodes,
        components=model_spec.get("components"),
        retrieve_fn=model_spec.get("retrieve_fn"),
        store_fn=model_spec.get("store_fn"),
        lr_fn=model_spec.get("lr_fn"),
        update_fn=model_spec.get("update_fn"),
        alpha=model_spec.get("alpha", None),
    )

    rt_col = model_spec.get("rt_col", "reaction_time")
    if rt_col in behavior.columns:
        df[rt_col] = pd.to_numeric(behavior[rt_col], errors="coerce").to_numpy(float)
    else:
        df[rt_col] = np.nan

    meta = dict(sub_id=sub_id, model=model_spec.get("name", "model"))
    return df, state, meta


def compute_readouts(df, state, meta, *, model_spec: Mapping[str, Any]):
    out = dict(**meta, n_trials=int(len(df)))

    r_ent, r_neff = rt_correlations(df, rt_col=model_spec.get("rt_col", "reaction_time"))
    out["rt_r_entropy"] = r_ent
    out["rt_r_neff"] = r_neff

    if model_spec.get("kappa") is not None:
        slope, r = confusability_vs_distance_slope(
            state,
            kappa=float(model_spec["kappa"]),
            same_char_bias=float(model_spec.get("same_char_bias", 0.0)),
            restrict=str(model_spec.get("restrict", "all")),
        )
    else:
        slope, r = np.nan, np.nan

    out["conf_slope"] = slope
    out["conf_r"] = r
    return out


def summarize_subjects(
    incl_subs,
    *,
    model_spec: Mapping[str, Any],
    verbose=True,
    on_error="skip",
    **fit_kwargs,
):
    rows = []
    n = len(incl_subs)

    for _, sub_id in tqdm(enumerate(incl_subs, start=1), total=n, desc="Subjects"):
        try:
            df, state, meta = fit_subject(sub_id, model_spec=model_spec, **fit_kwargs)
            if df is None:
                continue
            rows.append(compute_readouts(df, state, meta, model_spec=model_spec))

        except Exception as e:
            if on_error == "raise":
                raise
            if verbose:
                print(f"  skipped (error): {type(e).__name__}: {e}")
            continue

    df_sum = pd.DataFrame(rows)
    if not df_sum.empty:
        df_sum = df_sum.sort_values(["model", "sub_id"]).reset_index(drop=True)
    return df_sum


# ============================================================
# Simulation utilities
# ============================================================

def normalize(v, eps=1e-12):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < eps:
        return np.zeros_like(v)
    return v / n


def make_true_vectors(n_char):
    angles = np.linspace(0, 2 * np.pi, n_char, endpoint=False)
    return np.c_[np.cos(angles), np.sin(angles)]


def make_schedule(n_trials, n_char, mode="blocks", block_len=25, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if mode == "interleave":
        return rng.integers(0, n_char, size=n_trials)
    if mode == "blocks":
        seq = []
        c = 0
        while len(seq) < n_trials:
            seq.extend([c] * block_len)
            c = (c + 1) % n_char
        return np.asarray(seq[:n_trials], int)
    raise ValueError("mode must be 'blocks' or 'interleave'")


def choice_continuous(true_vec, noise_sd=0.25, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    v = np.asarray(true_vec, float) + rng.normal(0, noise_sd, size=2)
    return normalize(v)


def choice_discrete(true_vec, beta=7.0, n_dirs=8, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    angles = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    dirs = np.c_[np.cos(angles), np.sin(angles)]
    scores = beta * (dirs @ normalize(true_vec))
    scores = scores - scores.max()
    p = np.exp(scores)
    p = p / p.sum()
    idx = rng.choice(np.arange(n_dirs), p=p)
    return dirs[idx]


def choice_assimilation(true_vec, retrieved, lam=0.5, noise_sd=0.20, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    v = (1 - lam) * np.asarray(true_vec, float) + lam * np.asarray(retrieved, float)
    v = v + rng.normal(0, noise_sd, size=2)
    return normalize(v)


# ============================================================
# Simulation runner + snapshot
# ============================================================

def run_simulation(
    *,
    regime_name,
    model_spec: Mapping[str, Any],
    choice_mode="continuous",
    choice_kwargs=None,
    n_trials=300,
    n_char=3,
    schedule_mode="blocks",
    block_len=25,
    seed=0,
    t_switch=None,
    char_switch=None,
    new_vec=None,
):
    """Returns (df, base_true, state)."""
    choice_kwargs = {} if choice_kwargs is None else dict(choice_kwargs)
    rng = np.random.default_rng(seed)
    base_true = make_true_vectors(n_char)

    def true_vec_at_t(t, c):
        if (
            t_switch is not None
            and char_switch is not None
            and new_vec is not None
            and t >= int(t_switch)
            and c == int(char_switch)
        ):
            return np.asarray(new_vec, float)
        return np.asarray(base_true[c], float)

    state = model_spec["init_fn"](n_char)
    schedule = make_schedule(
        n_trials,
        n_char,
        mode=schedule_mode,
        block_len=block_len,
        rng=rng,
    )

    retr_fn = model_spec.get("retrieve_state_fn", None) or model_spec["retrieve_fn"]

    rows = []
    for t, c in enumerate(schedule):
        c = int(c)
        tv = true_vec_at_t(t, c)

        if choice_mode == "assimilation":
            retr, _ = retr_fn(state, c)
            x = choice_assimilation(tv, retr, rng=rng, **choice_kwargs)
        elif choice_mode == "continuous":
            x = choice_continuous(tv, rng=rng, **choice_kwargs)
        elif choice_mode == "discrete":
            x = choice_discrete(tv, rng=rng, **choice_kwargs)
        else:
            raise ValueError(
                "choice_mode must be 'continuous', 'discrete', or 'assimilation'"
            )

        state, out = model_spec["step_fn"](state, c, x)
        out = fill_retrieval_fields(out)

        rows.append(
            {
                "regime": regime_name,
                "model": model_spec.get("name", "model"),
                "t": int(t),
                "char": c,
                "x_affil": float(x[0]),
                "x_power": float(x[1]),
                "true_affil": float(tv[0]),
                "true_power": float(tv[1]),
                "mu_affil": float(out["affil_mean"]),
                "mu_power": float(out["power_mean"]),
                "retr_affil": float(out["retrieved_affil"]),
                "retr_power": float(out["retrieved_power"]),
                "alpha_used": float(out["alpha"]) if np.isfinite(out["alpha"]) else np.nan,
                "surprise": float(out["surprise"])
                if np.isfinite(out["surprise"])
                else np.nan,
                "entropy": float(out["retrieval_entropy"])
                if np.isfinite(out["retrieval_entropy"])
                else np.nan,
                "n_eff": float(out["retrieval_neff"])
                if np.isfinite(out["retrieval_neff"])
                else np.nan,
                "other_mass": float(out["other_mass"])
                if np.isfinite(out["other_mass"])
                else np.nan,
                "mem_n": int(out["mem_n"])
                if np.isfinite(out["mem_n"])
                else int(out.get("mem_n", 0)),
            }
        )

    df = pd.DataFrame(rows)
    df["err_to_true"] = np.sqrt(
        (df["mu_affil"] - df["true_affil"]) ** 2 + (df["mu_power"] - df["true_power"]) ** 2
    )
    df["retr_err_to_true"] = np.sqrt(
        (df["retr_affil"] - df["true_affil"]) ** 2
        + (df["retr_power"] - df["true_power"]) ** 2
    )
    return df, base_true, state


def snapshot_attention(
    *,
    model_spec: Mapping[str, Any],
    snapshot_t=220,
    cue_char=0,
    seed=0,
    topk=20,
    n_char=3,
    n_trials=360,
    noise_sd=0.25,
    schedule_mode="blocks",
    block_len=25,
):
    rng = np.random.default_rng(seed)
    base_true = make_true_vectors(n_char)
    state = model_spec["init_fn"](n_char)
    schedule = make_schedule(
        n_trials, n_char, mode=schedule_mode, block_len=block_len, rng=rng
    )
    retr_fn = model_spec.get("retrieve_state_fn", None) or model_spec["retrieve_fn"]

    for t, c in enumerate(schedule):
        tv = base_true[int(c)]
        x = choice_continuous(tv, noise_sd=noise_sd, rng=rng)
        state, _ = model_spec["step_fn"](state, int(c), x)

        if t == snapshot_t:
            retrieved, _info = retr_fn(state, int(cue_char))
            w, idx = model_spec["weights_fn"](state, cue_char=int(cue_char))

            print(f"Snapshot at t={snapshot_t}, cue_char={cue_char}, retrieved={retrieved}")

            if w is None or idx is None:
                print("No weights available for this model (no episodic memory).")
                return

            mem_chars_all = np.asarray(state.get("mem_chars", []), int)
            mem_chars = mem_chars_all[idx]
            order = np.argsort(w)[::-1][:topk]

            top = pd.DataFrame(
                {
                    "rank": np.arange(1, len(order) + 1),
                    "episode_idx": idx[order],
                    "episode_char": mem_chars[order],
                    "weight": w[order],
                }
            )
            top["cum_weight"] = np.cumsum(top["weight"])
            print(top.to_string(index=False))

            import matplotlib.pyplot as plt

            labels = [f"c{cc}:e{ee}" for cc, ee in zip(top["episode_char"], top["episode_idx"])]
            plt.figure(figsize=(10, 4))
            plt.bar(np.arange(len(top)), top["weight"].values)
            plt.xticks(np.arange(len(top)), labels, rotation=90)
            plt.ylabel("attention weight")
            plt.title("Top retrieved episodes (attention over memory)")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(6, 4))
            plt.plot(
                np.arange(1, len(top) + 1),
                top["cum_weight"].values,
                marker="o",
                linewidth=1.5,
            )
            plt.xlabel("top-k episodes included")
            plt.ylabel("cumulative weight")
            plt.title("Cumulative attention mass in top-k")
            plt.tight_layout()
            plt.show()
            return

    print("No snapshot captured (snapshot_t out of range).")


__all__ = [
    "ModelComponents",
    "build_model_spec",
    "choice_assimilation",
    "choice_continuous",
    "choice_discrete",
    "compose_model_spec",
    "compute_readouts",
    "confusability_vs_distance_slope",
    "confusion_prob_offdiag",
    "delta_rule_update",
    "distance_matrix_from_mu",
    "encode_characters",
    "fill_retrieval_fields",
    "fit_subject",
    "init_proto_state",
    "linreg_slope",
    "lr_running_mean",
    "make_components",
    "make_episode_df",
    "make_model_spec_episodic",
    "make_model_spec_prototype",
    "make_schedule",
    "make_true_vectors",
    "normalize",
    "online_step",
    "pearson_r",
    "predicted_confusability_mass",
    "retrieval_weights",
    "retrieve_none",
    "retrieve_softmax_episode",
    "rt_correlations",
    "run_online_model_functional",
    "run_simulation",
    "snapshot_attention",
    "store_mu_and_episode",
    "store_mu_only",
    "summarize_subjects",
]

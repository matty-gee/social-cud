import gc
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import load_img, concat_imgs, index_img


# ============================= helpers =============================

def get_rp24(confound_file):
    confounds = pd.read_csv(confound_file, sep="\t")

    base_cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    deriv_cols = [f"{c}_derivative1" for c in base_cols]
    power2_cols = [f"{c}_power2" for c in base_cols]
    deriv2_cols = [f"{c}_derivative1_power2" for c in base_cols]
    rp24_cols = base_cols + deriv_cols + power2_cols + deriv2_cols

    missing = [c for c in rp24_cols if c not in confounds.columns]
    if missing:
        raise ValueError(f"missing rp-24 columns: {missing}")

    rp_24 = confounds[rp24_cols].copy()
    rp_24 = rp_24.fillna(0.0)
    rp_24 = rp_24.astype(np.float32)
    assert rp_24.shape[1] == 24, f"expected 24 columns in rp_24, got {rp_24.shape[1]}"
    return rp_24

def build_trials_table(timing_file, behavior_file):
    """
    Build a trial table for nilearn GLM.

    Rules:
    - Keep only Narrative and Decision trials
    - Use timing file onsets
    - Narrative duration = timing-file duration
    - Decision duration = reaction_time from behavior file
    - If no response or missing RT, duration = 0
    """
    timing = pd.read_excel(timing_file)
    timing = timing.sort_values("onset").reset_index(drop=True)

    timing = timing[timing["trial_type"].isin(["Narrative", "Decision"])].copy()
    timing = timing.reset_index(drop=True)

    timing["trial_num"] = pd.to_numeric(timing["trial_num"], errors="coerce")
    timing["onset"] = pd.to_numeric(timing["onset"], errors="coerce")
    timing["duration"] = pd.to_numeric(timing["duration"], errors="coerce")

    if "decision_num" not in timing.columns:
        timing["decision_num"] = np.nan
    else:
        timing["decision_num"] = pd.to_numeric(timing["decision_num"], errors="coerce")

    timing["duration_model"] = timing["duration"]

    beh = pd.read_excel(behavior_file)
    beh = beh.sort_values("decision_num").reset_index(drop=True)

    beh["decision_num"] = pd.to_numeric(beh["decision_num"], errors="coerce")
    beh["reaction_time"] = pd.to_numeric(beh["reaction_time"], errors="coerce")
    beh["reaction_time"] = beh["reaction_time"].fillna(0.0)

    if "responded" in beh.columns:
        if beh["responded"].dtype == object:
            beh["responded"] = beh["responded"].astype(str).str.upper().eq("TRUE")
        else:
            beh["responded"] = beh["responded"].astype(bool)
    else:
        beh["responded"] = True

    beh["duration_model"] = beh["reaction_time"]
    beh.loc[~beh["responded"], "duration_model"] = 0.0
    beh["duration_model"] = beh["duration_model"].fillna(0.0)

    rt_map = beh.set_index("decision_num")["duration_model"]

    is_dec = timing["trial_type"].eq("Decision")
    timing.loc[is_dec, "duration_model"] = timing.loc[is_dec, "decision_num"].map(rt_map)
    timing.loc[is_dec, "duration_model"] = timing.loc[is_dec, "duration_model"].fillna(0.0)

    timing["duration"] = timing["duration_model"]
    timing = timing.drop(columns=["duration_model"])

    assert timing["onset"].notna().all(), "Non-finite onsets found"
    assert timing["duration"].notna().all(), "Non-finite durations found"

    return timing

def make_lss_events(trials, target_decision_num):
    """
    Build the 3-regressor LSS events table:
      - target_decision: the one target decision trial
      - other_decision: all other decision trials
      - other_narrative: all narrative trials
    """
    events = trials[["onset", "duration", "trial_type", "decision_num"]].copy()
    events = events[events["trial_type"].isin(["Narrative", "Decision"])].copy()

    events.loc[events["trial_type"] == "Narrative", "trial_type"] = "other_narrative"
    events.loc[events["trial_type"] == "Decision", "trial_type"] = "other_decision"

    is_target = events["decision_num"].eq(target_decision_num)
    if is_target.sum() != 1:
        raise ValueError(
            f"Expected exactly 1 target decision for decision_num={target_decision_num}, "
            f"found {is_target.sum()}"
        )

    events.loc[is_target, "trial_type"] = "target_decision"
    return events[["onset", "duration", "trial_type"]].copy()

def make_lss_events_for_trial(trials, target_model_trial_idx):
    """
    Build a 3-regressor LSS events table for one modeled trial:
      - target_decision or target_narrative: the one target trial
      - other_decision: all non-target decision trials
      - other_narrative: all non-target narrative trials
    """
    if "model_trial_idx" not in trials.columns:
        raise ValueError("trials must include model_trial_idx")

    events = trials[["onset", "duration", "trial_type", "model_trial_idx"]].copy()
    events = events[events["trial_type"].isin(["Narrative", "Decision"])].copy()

    is_target = events["model_trial_idx"].eq(target_model_trial_idx)
    if is_target.sum() != 1:
        raise ValueError(
            f"Expected exactly 1 target trial for model_trial_idx={target_model_trial_idx}, "
            f"found {is_target.sum()}"
        )

    target_type = events.loc[is_target, "trial_type"].iloc[0].lower()
    events.loc[events["trial_type"] == "Narrative", "trial_type"] = "other_narrative"
    events.loc[events["trial_type"] == "Decision", "trial_type"] = "other_decision"
    events.loc[is_target, "trial_type"] = f"target_{target_type}"

    return events[["onset", "duration", "trial_type"]].copy()

def force_nrows(df, n_rows):
    """Truncate or zero-pad confounds to match n_scans."""
    df = df.copy().reset_index(drop=True)
    if len(df) == n_rows:
        return df
    if len(df) > n_rows:
        return df.iloc[:n_rows].reset_index(drop=True)

    pad = pd.DataFrame(0.0, index=np.arange(n_rows - len(df)), columns=df.columns)
    return pd.concat([df, pad], axis=0, ignore_index=True)

def safe_delete(path):
    path = Path(path)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

def file_is_good(path, min_bytes=1024, check_load=True):
    """
    Basic file check:
    - exists
    - non-trivial size
    - can be opened as a NIfTI (optional)
    """
    path = Path(path)
    if not path.exists() or not path.is_file():
        return False

    try:
        if path.stat().st_size < min_bytes:
            return False
    except Exception:
        return False

    if check_load:
        try:
            img = load_img(str(path))
            _ = img.shape
            del img
        except Exception:
            return False

    return True

def find_first_existing(paths, require_good=False):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            if require_good:
                if file_is_good(p, check_load=True):
                    return p
            else:
                return p
    return None

def any_existing(paths):
    return any(Path(p).exists() for p in paths if p is not None)

def delete_all(paths):
    for p in paths:
        if p is not None:
            safe_delete(p)

def find_task_specific_mask_file(sub_id, func_dir, anat_dir):
    """
    Prefer the task-specific functional-space brain mask.
    Fall back to broader patterns only if needed.
    """
    candidates = [
        next(func_dir.glob(f"{sub_id}_task-socialnav_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii*"), None),
        next(func_dir.glob(f"{sub_id}_task-socialnav*_desc-brain_mask.nii*"), None),
        next(func_dir.glob(f"{sub_id}*_task-socialnav*_desc-brain_mask.nii*"), None),
        next(anat_dir.glob(f"{sub_id}_desc-brain_mask.nii*"), None),
        next(anat_dir.glob(f"{sub_id}*_desc-brain_mask.nii*"), None),
    ]
    return find_first_existing(candidates, require_good=False)

def copy_mask_to_output(mask_file, out_dir, out_name="mask.nii.gz"):
    """
    Copy the chosen mask into the subject output directory and return the copied path.
    Always writes as .nii.gz, regardless of original extension.
    """
    mask_file = Path(mask_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_mask = out_dir / out_name

    # remove any stale copy first
    if out_mask.exists():
        safe_delete(out_mask)

    img = load_img(str(mask_file))
    img.to_filename(str(out_mask))

    if not file_is_good(out_mask):
        raise RuntimeError(f"Copied mask failed validation: {out_mask}")

    return out_mask

def get_trial_beta_path(trialmaps_dir, trial_idx, trial_type="decision"):
    trial_type = str(trial_type).lower()
    return trialmaps_dir / f"{trial_type}_{trial_idx:04d}_beta.nii.gz"

def get_n_vols(path):
    shape = load_img(str(path)).shape
    return 1 if len(shape) == 3 else shape[3]

def merged_file_is_good(path, expected_n_vols):
    if not file_is_good(path):
        return False
    try:
        return get_n_vols(path) == expected_n_vols
    except Exception:
        return False

def find_good_merged(paths, expected_n_vols):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists() and merged_file_is_good(p, expected_n_vols):
            return p
    return None

def split_4d_to_trialmaps(merged_file, out_files):
    """
    Extract missing 3D trial maps from an existing 4D beta file.
    This lets restarts reuse old merged outputs without rerunning GLMs.
    """
    merged_file = Path(merged_file)
    out_files = [Path(p) for p in out_files]

    if not merged_file_is_good(merged_file, len(out_files)):
        raise RuntimeError(f"Cannot split invalid merged file: {merged_file}")

    img = load_img(str(merged_file))
    try:
        for ii, out_file in enumerate(out_files):
            if file_is_good(out_file):
                continue
            if out_file.exists():
                safe_delete(out_file)

            vol = index_img(img, ii)
            vol.to_filename(str(out_file))
            del vol

            if not file_is_good(out_file):
                raise RuntimeError(f"Extracted trial map failed validation: {out_file}")
    finally:
        del img
        gc.collect()

def ensure_merged_output(img_files, out_file, variants):
    """
    Ensure out_file exists as a valid 4D merge of img_files.
    If another valid variant already exists, convert it instead of re-merging.
    """
    out_file = Path(out_file)
    variants = [Path(p) for p in variants]
    expected_n_vols = len(img_files)

    if merged_file_is_good(out_file, expected_n_vols):
        return out_file

    existing = find_good_merged([p for p in variants if p != out_file], expected_n_vols)
    if existing is not None:
        if out_file.exists():
            safe_delete(out_file)
        img = load_img(str(existing))
        img.to_filename(str(out_file))
        del img
        gc.collect()

        if not merged_file_is_good(out_file, expected_n_vols):
            raise RuntimeError(f"Converted merged output failed validation: {out_file}")
        return out_file

    safe_delete(out_file)
    merge_trialmaps_to_4d(img_files, out_file)
    return out_file

def merge_trialmaps_to_4d(img_files, out_file):
    if len(img_files) == 0:
        raise ValueError(f"No images provided for merge: {out_file}")

    missing = [str(p) for p in img_files if not file_is_good(p)]
    if missing:
        raise FileNotFoundError("Missing or invalid trial maps before merge:\n" + "\n".join(missing))

    merged = concat_imgs([str(p) for p in img_files], auto_resample=False)
    merged.to_filename(str(out_file))

    merged_shape = load_img(str(out_file)).shape
    n_vols = 1 if len(merged_shape) == 3 else merged_shape[3]
    if n_vols != len(img_files):
        raise RuntimeError(
            f"Merged file has wrong number of volumes: expected {len(img_files)}, found {n_vols}"
        )

    del merged
    gc.collect()

def _fit_lss_trial_beta(
    sub_id,
    trial_type,
    type_trial_idx,
    model_trial_idx,
    trials,
    func_file,
    out_mask_file,
    rp_24,
    beta_file,
    tr,
    slice_time_ref,
    high_pass,
    glm_n_jobs,
    nilearn_memory,
    memory_level,
    max_fit_attempts,
):
    beta_file = Path(beta_file)
    target_col = f"target_{str(trial_type).lower()}"

    if file_is_good(beta_file):
        return {
            "trial_type": trial_type,
            "type_trial_idx": type_trial_idx,
            "model_trial_idx": model_trial_idx,
            "beta_file": str(beta_file),
            "status": "reused",
        }

    max_fit_attempts = max(1, int(max_fit_attempts))
    last_error = None

    for attempt in range(1, max_fit_attempts + 1):
        if beta_file.exists():
            safe_delete(beta_file)

        glm = None
        design = None
        beta_img = None
        events_lss = None

        try:
            events_lss = make_lss_events_for_trial(trials, model_trial_idx)

            glm = FirstLevelModel(
                t_r=tr,
                slice_time_ref=slice_time_ref,
                hrf_model="spm",
                drift_model="cosine",
                high_pass=high_pass,
                mask_img=str(out_mask_file),
                smoothing_fwhm=None,
                memory=nilearn_memory,
                memory_level=memory_level,
                signal_scaling=False,
                n_jobs=glm_n_jobs,
                minimize_memory=True,
            )

            glm = glm.fit(
                run_imgs=str(func_file),
                events=events_lss,
                confounds=rp_24,
            )

            design = glm.design_matrices_[0]
            if target_col not in design.columns:
                raise RuntimeError(
                    f"'{target_col}' missing from design columns:\n{design.columns.tolist()}"
                )

            beta_img = glm.compute_contrast(target_col, output_type="effect_size")
            beta_img.to_filename(str(beta_file))

            if not file_is_good(beta_file):
                raise RuntimeError("Saved per-trial beta output failed file check")

            return {
                "trial_type": trial_type,
                "type_trial_idx": type_trial_idx,
                "model_trial_idx": model_trial_idx,
                "beta_file": str(beta_file),
                "status": "fit",
                "attempts": attempt,
            }

        except Exception as exc:
            last_error = exc
            safe_delete(beta_file)
            if attempt >= max_fit_attempts:
                raise

        finally:
            if beta_img is not None:
                del beta_img
            if design is not None:
                del design
            if glm is not None:
                del glm
            if events_lss is not None:
                del events_lss
            gc.collect()

    raise RuntimeError(
        f"Failed to fit {trial_type} trial {type_trial_idx} after "
        f"{max_fit_attempts} attempts"
    ) from last_error

# ============================= main runner =============================

def run_lss_subject(
    sub_id,
    preprc_dir,
    behav_dir,
    timing_file,
    glm_dir,
    tr=1.0,
    slice_time_ref=0.5,
    high_pass=1/128,
    n_jobs=1,
    parallel_backend="loky",
    glm_n_jobs=1,
    nilearn_memory=None,
    memory_level=1,
    max_fit_attempts=2,
):
    """
    Runs decision/narrative-trial LSS for one subject, beta-only.

    Restart logic:
    - Reuse any valid cached per-trial beta maps.
    - If a valid merged decision/narrative output exists but per-trial maps are missing,
      split the merged output into per-trial maps instead of rerunning GLMs.
    - Fit only missing per-trial beta maps.
    - Write decision_trials_beta.nii, narrative_trials_beta.nii, and all_trials_beta.nii.

    Parallelization:
    - n_jobs parallelizes across missing trial-level GLMs.
    - glm_n_jobs is passed through to Nilearn's FirstLevelModel for each GLM.
    - Avoid setting both n_jobs and glm_n_jobs high unless enough CPUs/RAM are allocated.
    - max_fit_attempts retries a missing trial GLM after deleting any partial output.
    """
    print(f"\n[{sub_id}] starting LSS beta-only run", flush=True)

    subj_dir = Path(preprc_dir) / sub_id
    func_dir = subj_dir / "func"
    anat_dir = subj_dir / "anat"

    out_dir = Path(glm_dir) / sub_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = out_dir / "_tmp_lss_decision"
    trialmaps_dir = tmp_root / "trialmaps_3d"
    trialmaps_dir.mkdir(parents=True, exist_ok=True)

    if nilearn_memory == "auto":
        nilearn_memory = str(tmp_root / "nilearn_cache")

    # remove legacy per-trial t maps if they exist from older runs
    for old_t in trialmaps_dir.glob("decision_*_t.nii*"):
        safe_delete(old_t)

    merged_decision_nii = out_dir / "decision_trials_beta.nii"
    merged_decision_gz = out_dir / "decision_trials_beta.nii.gz"
    merged_decision_variants = [merged_decision_nii, merged_decision_gz]

    merged_narrative_nii = out_dir / "narrative_trials_beta.nii"
    merged_narrative_gz = out_dir / "narrative_trials_beta.nii.gz"
    merged_narrative_variants = [merged_narrative_nii, merged_narrative_gz]

    merged_all_nii = out_dir / "all_trials_beta.nii"
    merged_all_gz = out_dir / "all_trials_beta.nii.gz"
    merged_all_variants = [merged_all_nii, merged_all_gz]

    # remove legacy merged t maps if they exist from older runs
    legacy_t_variants = [
        out_dir / "decision_trials_t.nii.gz",
        out_dir / "decision_trials_t.nii",
    ]
    delete_all([p for p in legacy_t_variants if p.exists()])

    # -------------------- find files --------------------
    behav_file = next(Path(behav_dir).glob(f"{sub_id}.xlsx"), None)
    func_file = next(func_dir.glob(f"{sub_id}_task-socialnav*_desc-preproc_bold.nii*"), None)
    anat_file = next(anat_dir.glob(f"{sub_id}*_desc-preproc_T1w.nii*"), None)
    mask_file = find_task_specific_mask_file(sub_id, func_dir, anat_dir)
    confound_file = next(func_dir.glob(f"{sub_id}_task-socialnav*_desc-confounds_timeseries.tsv"), None)

    if behav_file is None:
        raise FileNotFoundError(f"{sub_id}: could not find behavioral file in {behav_dir}")
    if func_file is None:
        raise FileNotFoundError(f"{sub_id}: could not find func file in {func_dir}")
    if anat_file is None:
        raise FileNotFoundError(f"{sub_id}: could not find anat file in {anat_dir}")
    if confound_file is None:
        raise FileNotFoundError(f"{sub_id}: could not find confounds file in {func_dir}")
    if mask_file is None:
        raise FileNotFoundError(
            f"{sub_id}: could not find task-specific brain mask in {func_dir} "
            f"(or fallback mask in {anat_dir})"
        )

    # copy mask into subject output folder as mask.nii.gz
    out_mask_file = copy_mask_to_output(mask_file, out_dir, out_name="mask.nii.gz")

    print(f"[{sub_id}] behav:     {behav_file}", flush=True)
    print(f"[{sub_id}] func:      {func_file}", flush=True)
    print(f"[{sub_id}] anat:      {anat_file}", flush=True)
    print(f"[{sub_id}] mask src:  {mask_file}", flush=True)
    print(f"[{sub_id}] mask out:  {out_mask_file}", flush=True)
    print(f"[{sub_id}] confounds: {confound_file}", flush=True)

    # -------------------- confounds --------------------
    rp_24 = get_rp24(confound_file)
    n_scans = load_img(str(func_file)).shape[-1]
    print(f"[{sub_id}] n_scans: {n_scans}", flush=True)

    if len(rp_24) != n_scans:
        print(f"[{sub_id}] confounds rows ({len(rp_24)}) != n_scans ({n_scans}) -> trunc/pad", flush=True)
        rp_24 = force_nrows(rp_24, n_scans)

    # -------------------- trials --------------------
    trials = build_trials_table(timing_file, behav_file)
    trials = trials.reset_index(drop=True)
    trials.insert(0, "model_trial_idx", np.arange(1, len(trials) + 1))

    decision_trials = trials.loc[
        trials["trial_type"] == "Decision",
        ["model_trial_idx", "trial_num", "decision_num", "onset", "duration"]
    ].copy().reset_index(drop=True)
    decision_trials.insert(0, "type_trial_idx", np.arange(1, len(decision_trials) + 1))

    narrative_trials = trials.loc[
        trials["trial_type"] == "Narrative",
        ["model_trial_idx", "trial_num", "decision_num", "onset", "duration"]
    ].copy().reset_index(drop=True)
    narrative_trials.insert(0, "type_trial_idx", np.arange(1, len(narrative_trials) + 1))

    n_dec = len(decision_trials)
    if n_dec == 0:
        raise RuntimeError(f"{sub_id}: no decision trials found")
    n_narr = len(narrative_trials)
    if n_narr == 0:
        raise RuntimeError(f"{sub_id}: no narrative trials found")

    print(f"[{sub_id}] found {n_dec} decision trials", flush=True)
    print(f"[{sub_id}] found {n_narr} narrative trials", flush=True)

    # save the exact order used for per-trial output naming
    decision_index_file = out_dir / "decision_trial_index.tsv"
    decision_trials_for_save = decision_trials.copy()
    decision_trials_for_save.insert(0, "lss_trial_idx", decision_trials_for_save["type_trial_idx"])
    decision_trials_for_save.to_csv(decision_index_file, sep="\t", index=False)

    narrative_index_file = out_dir / "narrative_trial_index.tsv"
    narrative_trials_for_save = narrative_trials.copy()
    narrative_trials_for_save.insert(0, "lss_trial_idx", narrative_trials_for_save["type_trial_idx"])
    narrative_trials_for_save.to_csv(narrative_index_file, sep="\t", index=False)

    trials["type_trial_idx"] = trials.groupby("trial_type").cumcount() + 1
    trials["beta_file"] = [
        get_trial_beta_path(trialmaps_dir, row.type_trial_idx, row.trial_type).name
        for row in trials.itertuples(index=False)
    ]

    all_index_file = out_dir / "all_trial_index.tsv"
    all_trials_for_save = trials.copy()
    all_trials_for_save.insert(0, "all_trial_idx", all_trials_for_save["model_trial_idx"])
    all_trials_for_save.to_csv(all_index_file, sep="\t", index=False)

    decision_beta_files = [
        get_trial_beta_path(trialmaps_dir, ii, "decision")
        for ii in decision_trials["type_trial_idx"].tolist()
    ]
    narrative_beta_files = [
        get_trial_beta_path(trialmaps_dir, ii, "narrative")
        for ii in narrative_trials["type_trial_idx"].tolist()
    ]
    all_beta_files = [
        get_trial_beta_path(trialmaps_dir, row.type_trial_idx, row.trial_type)
        for row in trials.itertuples(index=False)
    ]

    # If old merged outputs exist but 3D maps are missing, split them instead of rerunning GLMs.
    all_merged_good = find_good_merged(merged_all_variants, len(all_beta_files))
    if all_merged_good is not None:
        print(f"[{sub_id}] reusing merged all-trials output: {all_merged_good}", flush=True)
        split_4d_to_trialmaps(all_merged_good, all_beta_files)

    decision_merged_good = find_good_merged(merged_decision_variants, n_dec)
    if decision_merged_good is not None:
        print(f"[{sub_id}] reusing merged decision output: {decision_merged_good}", flush=True)
        split_4d_to_trialmaps(decision_merged_good, decision_beta_files)

    narrative_merged_good = find_good_merged(merged_narrative_variants, n_narr)
    if narrative_merged_good is not None:
        print(f"[{sub_id}] reusing merged narrative output: {narrative_merged_good}", flush=True)
        split_4d_to_trialmaps(narrative_merged_good, narrative_beta_files)

    # -------------------- fit only missing per-trial betas --------------------
    fit_jobs = []
    n_reused = 0

    for row in decision_trials.itertuples(index=False):
        beta_file = get_trial_beta_path(trialmaps_dir, row.type_trial_idx, "decision")
        if file_is_good(beta_file):
            n_reused += 1
            print(f"[{sub_id}] reuse decision trial {row.type_trial_idx:03d}/{n_dec}", flush=True)
            continue

        fit_jobs.append({
            "sub_id": sub_id,
            "trial_type": "decision",
            "type_trial_idx": int(row.type_trial_idx),
            "model_trial_idx": int(row.model_trial_idx),
            "trials": trials,
            "func_file": str(func_file),
            "out_mask_file": str(out_mask_file),
            "rp_24": rp_24,
            "beta_file": str(beta_file),
            "tr": tr,
            "slice_time_ref": slice_time_ref,
            "high_pass": high_pass,
            "glm_n_jobs": glm_n_jobs,
            "nilearn_memory": nilearn_memory,
            "memory_level": memory_level,
            "max_fit_attempts": max_fit_attempts,
        })

    for row in narrative_trials.itertuples(index=False):
        beta_file = get_trial_beta_path(trialmaps_dir, row.type_trial_idx, "narrative")
        if file_is_good(beta_file):
            n_reused += 1
            print(f"[{sub_id}] reuse narrative trial {row.type_trial_idx:03d}/{n_narr}", flush=True)
            continue

        fit_jobs.append({
            "sub_id": sub_id,
            "trial_type": "narrative",
            "type_trial_idx": int(row.type_trial_idx),
            "model_trial_idx": int(row.model_trial_idx),
            "trials": trials,
            "func_file": str(func_file),
            "out_mask_file": str(out_mask_file),
            "rp_24": rp_24,
            "beta_file": str(beta_file),
            "tr": tr,
            "slice_time_ref": slice_time_ref,
            "high_pass": high_pass,
            "glm_n_jobs": glm_n_jobs,
            "nilearn_memory": nilearn_memory,
            "memory_level": memory_level,
            "max_fit_attempts": max_fit_attempts,
        })

    if fit_jobs:
        print(f"[{sub_id}] fitting {len(fit_jobs)} missing trial betas with n_jobs={n_jobs}", flush=True)
        if int(n_jobs) == 1:
            results = []
            for job in fit_jobs:
                print(
                    f"[{sub_id}] fit   {job['trial_type']} trial {job['type_trial_idx']:03d}",
                    flush=True,
                )
                results.append(_fit_lss_trial_beta(**job))
        else:
            results = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
                delayed(_fit_lss_trial_beta)(**job) for job in fit_jobs
            )

        for result in results:
            attempt_msg = ""
            if result.get("attempts", 1) > 1:
                attempt_msg = f" after {result['attempts']} attempts"
            print(
                f"[{sub_id}] saved {result['trial_type']} trial "
                f"{result['type_trial_idx']:03d} beta{attempt_msg}",
                flush=True,
            )
    else:
        results = []

    n_fit = sum(1 for result in results if result["status"] == "fit")
    print(f"[{sub_id}] finished trial loop | reused={n_reused}, fit={n_fit}", flush=True)

    # -------------------- verify all per-trial betas exist --------------------
    missing_beta = [str(p) for p in decision_beta_files + narrative_beta_files if not file_is_good(p)]
    if missing_beta:
        raise RuntimeError("Missing beta maps:\n" + "\n".join(missing_beta))

    print(f"[{sub_id}] verified {len(decision_beta_files)} decision trial beta maps", flush=True)
    print(f"[{sub_id}] verified {len(narrative_beta_files)} narrative trial beta maps", flush=True)

    # -------------------- merge to final 4D beta outputs --------------------
    print(f"[{sub_id}] ensuring decision merge -> {merged_decision_nii.name}", flush=True)
    ensure_merged_output(decision_beta_files, merged_decision_nii, merged_decision_variants)

    print(f"[{sub_id}] ensuring narrative merge -> {merged_narrative_nii.name}", flush=True)
    ensure_merged_output(narrative_beta_files, merged_narrative_nii, merged_narrative_variants)

    print(f"[{sub_id}] ensuring chronological all-trials merge -> {merged_all_nii.name}", flush=True)
    ensure_merged_output(all_beta_files, merged_all_nii, merged_all_variants)

    print(f"[{sub_id}] done", flush=True)
    print(f"[{sub_id}] decision beta:  {merged_decision_nii}", flush=True)
    print(f"[{sub_id}] narrative beta: {merged_narrative_nii}", flush=True)
    print(f"[{sub_id}] all beta:       {merged_all_nii}", flush=True)

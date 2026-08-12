import gc
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import load_img, concat_imgs


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

def get_trial_beta_path(trialmaps_dir, trial_idx):
    return trialmaps_dir / f"decision_{trial_idx:04d}_beta.nii.gz"

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
):
    """
    Runs decision-trial LSS for one subject, beta-only.

    Restart logic:
    - If merged beta output already exists (.nii.gz or .nii) -> skip subject.
    - If merged beta output does not exist -> reuse any valid cached per-trial beta maps.
    - Per-trial beta maps are kept after success so restarts can resume or re-merge.
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

    # remove legacy per-trial t maps if they exist from older runs
    for old_t in trialmaps_dir.glob("decision_*_t.nii*"):
        safe_delete(old_t)

    merged_beta_gz = out_dir / "decision_trials_beta.nii.gz"
    merged_beta_nii = out_dir / "decision_trials_beta.nii"
    merged_beta_variants = [merged_beta_gz, merged_beta_nii]

    # remove legacy merged t maps if they exist from older runs
    legacy_t_variants = [
        out_dir / "decision_trials_t.nii.gz",
        out_dir / "decision_trials_t.nii",
    ]
    delete_all([p for p in legacy_t_variants if p.exists()])

    beta_final_good = find_first_existing(merged_beta_variants, require_good=True)

    # -------------------- done? --------------------
    if beta_final_good is not None:
        print(f"[{sub_id}] merged beta output already exists -> skipping", flush=True)
        print(f"  beta: {beta_final_good}", flush=True)
        return

    # if partial or invalid final beta exists, delete and rebuild from per-trial maps
    if any_existing(merged_beta_variants):
        print(f"[{sub_id}] partial or invalid merged beta found -> deleting final-output variants and rebuilding", flush=True)
        delete_all(merged_beta_variants)

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
    decision_trials = trials.loc[
        trials["trial_type"] == "Decision",
        ["trial_num", "decision_num", "onset", "duration"]
    ].copy().reset_index(drop=True)

    n_dec = len(decision_trials)
    if n_dec == 0:
        raise RuntimeError(f"{sub_id}: no decision trials found")

    print(f"[{sub_id}] found {n_dec} decision trials", flush=True)

    # save the exact order used for per-trial output naming
    decision_index_file = out_dir / "decision_trial_index.tsv"
    decision_trials_for_save = decision_trials.copy()
    decision_trials_for_save.insert(0, "lss_trial_idx", np.arange(1, len(decision_trials_for_save) + 1))
    decision_trials_for_save.to_csv(decision_index_file, sep="\t", index=False)

    expected_beta_files = []
    n_reused = 0
    n_fit = 0

    # -------------------- loop over decision trials --------------------
    print(f"[{sub_id}] starting trial loop", flush=True)

    for ii, target_decision_num in enumerate(decision_trials["decision_num"].tolist(), start=1):
        beta_file = get_trial_beta_path(trialmaps_dir, ii)
        expected_beta_files.append(beta_file)

        # reuse complete existing per-trial beta
        if file_is_good(beta_file):
            n_reused += 1
            print(f"[{sub_id}] reuse trial {ii:03d}/{n_dec} (decision_num={target_decision_num})", flush=True)
            continue

        # if partial or bad trial output exists, delete and rerun
        if beta_file.exists():
            safe_delete(beta_file)

        print(f"[{sub_id}] fit   trial {ii:03d}/{n_dec} (decision_num={target_decision_num})", flush=True)

        glm = None
        design = None
        beta_img = None
        events_lss = None

        try:
            events_lss = make_lss_events(trials, target_decision_num)

            glm = FirstLevelModel(
                t_r=tr,
                slice_time_ref=slice_time_ref,
                hrf_model="spm",
                drift_model="cosine",
                high_pass=high_pass,
                mask_img=str(out_mask_file),
                smoothing_fwhm=None,
                signal_scaling=False,
                minimize_memory=True,
            )

            glm = glm.fit(
                run_imgs=str(func_file),
                events=events_lss,
                confounds=rp_24,
            )

            design = glm.design_matrices_[0]
            if "target_decision" not in design.columns:
                raise RuntimeError(
                    f"'target_decision' missing from design columns:\n{design.columns.tolist()}"
                )

            beta_img = glm.compute_contrast("target_decision", output_type="effect_size")
            beta_img.to_filename(str(beta_file))

            if not file_is_good(beta_file):
                raise RuntimeError("Saved per-trial beta output failed file check")

            n_fit += 1
            print(f"[{sub_id}] saved trial {ii:03d}/{n_dec} beta", flush=True)

        except Exception:
            safe_delete(beta_file)
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

    print(f"[{sub_id}] finished trial loop | reused={n_reused}, fit={n_fit}", flush=True)

    # -------------------- verify all per-trial betas exist --------------------
    missing_beta = [str(p) for p in expected_beta_files if not file_is_good(p)]
    if missing_beta:
        raise RuntimeError("Missing beta maps:\n" + "\n".join(missing_beta))

    print(f"[{sub_id}] verified {len(expected_beta_files)} trial beta maps", flush=True)

    # -------------------- merge to final 4D beta output --------------------
    safe_delete(merged_beta_gz)
    safe_delete(merged_beta_nii)

    print(f"[{sub_id}] merging beta maps -> {merged_beta_gz.name}", flush=True)
    merge_trialmaps_to_4d(expected_beta_files, merged_beta_gz)
    print(f"[{sub_id}] merge complete", flush=True)

    print(f"[{sub_id}] done", flush=True)
    print(f"[{sub_id}] merged beta: {merged_beta_gz}", flush=True)
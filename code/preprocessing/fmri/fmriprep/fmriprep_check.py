import os
import glob
from pathlib import Path

# Source of truth: subjects that exist in BIDS
BIDS_BASE = "/sc/arion/projects/OlfMem/mgs/data/BIDS"

# fMRIPrep output directory to check for completed outputs
FMRIPREP_BASE = "/sc/arion/projects/OlfMem/mgs/data/preprocessed/fmriprep/derivatives-fmap/fmriprep"

OUT_TSV = "/sc/arion/projects/OlfMem/mgs/code/fmriprep/participants.tsv"


def has_any(pattern: str) -> bool:
    return bool(glob.glob(pattern))


def main():
    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)

    # First: get all existing BIDS subjects
    bids_subject_dirs = sorted(
        d for d in glob.glob(os.path.join(BIDS_BASE, "sub-*"))
        if os.path.isdir(d)
    )

    incomplete_subjects = []
    missing_reasons = {}

    for bids_sub_dir in bids_subject_dirs:
        subject_id = os.path.basename(bids_sub_dir)  # e.g., sub-18001

        # Then: check whether this subject is complete in fMRIPrep
        func_dir = os.path.join(FMRIPREP_BASE, subject_id, "func")

        reasons = []

        if not os.path.isdir(func_dir):
            reasons.append("missing fmriprep func folder")
        else:
            social_pat = os.path.join(
                func_dir,
                f"{subject_id}_task-socialnav*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
            )
            rest_pat = os.path.join(
                func_dir,
                f"{subject_id}_task-rest*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
            )

            has_social = has_any(social_pat)
            has_rest = has_any(rest_pat)

            if not has_social:
                reasons.append("missing socialnav preproc bold")

            if not has_rest:
                reasons.append("missing rest preproc bold")

        # Keep only incomplete subjects
        if reasons:
            sub_id_no_prefix = subject_id.replace("sub-", "")
            incomplete_subjects.append(sub_id_no_prefix)
            missing_reasons[sub_id_no_prefix] = reasons

    with open(OUT_TSV, "w") as f:
        f.write("id\n")
        for sub_id in incomplete_subjects:
            f.write(f"{sub_id}\n")

    n_bids = len(bids_subject_dirs)
    n_incomplete = len(incomplete_subjects)
    n_complete = n_bids - n_incomplete

    print(f"Found {n_bids} BIDS subjects.")
    print(f"Complete in fMRIPrep: {n_complete}")
    print(f"Incomplete / missing in fMRIPrep: {n_incomplete}")
    print(f"\nWrote {n_incomplete} incomplete subjects to: {OUT_TSV}")

    if incomplete_subjects:
        print("\nIncomplete subjects:")
        for sub_id in incomplete_subjects:
            reason_str = "; ".join(missing_reasons[sub_id])
            print(f"{sub_id}: {reason_str}")
    else:
        print("\nNo incomplete subjects found.")


if __name__ == "__main__":
    main()
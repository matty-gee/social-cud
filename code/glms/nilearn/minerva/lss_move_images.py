from pathlib import Path
import shutil

SRC_ROOT = Path("/sc/arion/projects/OlfMem/mgs/analyses/lss_decision_fmriprep/glms")
DST_DIR  = Path("/sc/arion/projects/OlfMem/mgs/analyses/lss_decision_fmriprep/images")

DST_DIR.mkdir(parents=True, exist_ok=True)

for src in sorted(SRC_ROOT.glob("sub-*/decision_trials_beta.nii.gz")):
    sub_id = src.parent.name  # e.g., sub-18001
    dst = DST_DIR / f"{sub_id}_{src.name}"

    if dst.exists():
        print(f"Skipping existing file: {dst}")
        continue

    print(f"Moving {src} -> {dst}")
    shutil.move(str(src), str(dst))

print("Done.")
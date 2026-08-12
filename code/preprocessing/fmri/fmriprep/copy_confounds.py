import os
import glob
import shutil

BASE_DIR = "/sc/arion/projects/OlfMem/mgs/data/preprocessed/fmriprep/derivatives-fmap/fmriprep"
OUT_DIR = "/sc/arion/projects/OlfMem/mgs/data/quality-control/fmriprep_confounds"

# Create output folder if it does not exist
os.makedirs(OUT_DIR, exist_ok=True)

# Find all subject func folders
func_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "sub-*", "func")))

n_copied = 0
n_missing = 0

for func_dir in func_dirs:
    sub_id = os.path.basename(os.path.dirname(func_dir))

    # Find confounds files in this subject's func folder
    confound_files = sorted(
        glob.glob(os.path.join(func_dir, "*confounds_timeseries.tsv"))
    )

    if not confound_files:
        print(f"{sub_id}: no confounds file found")
        n_missing += 1
        continue

    for src in confound_files:
        fname = os.path.basename(src)
        dst = os.path.join(OUT_DIR, fname)

        # Copy, preserving metadata; original stays intact
        shutil.copy2(src, dst)
        print(f"{sub_id}: copied {fname}")
        n_copied += 1

print("\nDone.")
print(f"Copied files: {n_copied}")
print(f"Subjects with no confounds file: {n_missing}")
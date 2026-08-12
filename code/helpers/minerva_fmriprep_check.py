#!/usr/bin/env python3
import os
import glob

BASE = "/sc/arion/projects/OlfMem/mgs/data/preprocessed/fmriprep/derivatives-fmap/fmriprep"
OUT_TXT = "completed.txt"

def has_any(pattern: str) -> int:
    return 1 if glob.glob(pattern) else 0

def main():
    
    # Find all sub-*/func directories
    func_dirs = sorted(
        d for d in glob.glob(os.path.join(BASE, "sub-*", "func"))
        if os.path.isdir(d)
    )

    with open(OUT_TXT, "w") as f:
        f.write("func_dir\thas_socialnav\thas_rest\n")
        for func_dir in func_dirs:
            # Patterns: allow * between sub- and task to handle run/acq/etc.
            social_pat = os.path.join(
                func_dir,
                "sub-*_task-socialnav*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
            )
            rest_pat = os.path.join(
                func_dir,
                "sub-*_task-rest*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
            )

            has_social = has_any(social_pat)
            has_rest = has_any(rest_pat)

            f.write(f"{func_dir}\t{has_social}\t{has_rest}\n")

    print(f"Wrote: {OUT_TXT} ({len(func_dirs)} func dirs)")

if __name__ == "__main__":
    main()

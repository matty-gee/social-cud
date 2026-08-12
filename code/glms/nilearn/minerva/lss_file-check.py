#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import re

base_dir = Path("/sc/arion/projects/OlfMem/mgs/analyses/lss_decision/glms")

def get_largest_3d_beta_num(trialmaps_dir: Path) -> int:
    """
    Return the largest numeric image index among *.nii.gz files in trialmaps_dir.
    If no images are present, return 0.

    Example:
        decision_0001_beta.nii.gz -> 1
        decision_0013_beta.nii.gz -> 13
    """
    nii_files = sorted(trialmaps_dir.glob("*.nii.gz"))
    if not nii_files:
        return 0

    nums = []
    for f in nii_files:
        m = re.search(r"(\d+)", f.name)
        if m:
            nums.append(int(m.group(1)))

    return max(nums) if nums else 0


rows = []
for sub_dir in sorted(base_dir.glob("sub-*")):
    if not sub_dir.is_dir():
        continue

    beta_4d_file = sub_dir / "decision_trials_beta.nii.gz"
    beta_present = beta_4d_file.exists()

    if beta_present:
        num_3d_betas = 0
    else:
        trialmaps_dir = sub_dir / "_tmp_lss_decision" / "trialmaps_3d"
        num_3d_betas = get_largest_3d_beta_num(trialmaps_dir)

    row = {
        "sub_folder": sub_dir.name,
        "4d_beta_present": beta_present,
        "num_3d_betas": num_3d_betas,
    }
    rows.append(row)

    print(
        f"{row['sub_folder']}\t"
        f"4d_beta_present={row['4d_beta_present']}\t"
        f"num_3d_betas={row['num_3d_betas']}"
    )

df = pd.DataFrame(rows)

out_file = base_dir / "beta_presence_summary.xlsx"
df.to_excel(out_file, index=False)

print(f"\nWrote {len(df)} rows to: {out_file}")
"""
sitecustomize.py — global import hook for this repo.

WHAT THIS IS
------------
This file is automatically imported by Python at interpreter startup
(if it is importable). We use it to ensure that the repo's `code/`
directory is always on `sys.path`, so that modules like:

    from shared.main import ...

work from ANY notebook or script, regardless of the current working
directory or folder depth.

WHY THIS EXISTS
---------------
Notebooks in `code/analyses/...` run with a deep working directory,
which normally prevents Python from finding `code/shared/`.
This avoids per-notebook path hacks, editable installs, or packaging.

HOW IT WORKS
------------
- The conda environment sets:
      PYTHONPATH=/Users/matty_gee/Projects/SocialCUD/code
      SNT_PROJECT_ROOT=/Users/matty_gee/Desktop/Social/SocialCUD
- That makes this file importable at startup.
- Python auto-imports `sitecustomize`.
- We prepend the repo's `code/` directory to `sys.path`.

REQUIREMENT (ONE-TIME SETUP)
----------------------------
This file ONLY runs if the directory containing it is on PYTHONPATH.
To enable it:

    conda activate social_cud
    conda env config vars set \
        PYTHONPATH=/Users/matty_gee/Projects/SocialCUD/code \
        SNT_PROJECT_ROOT=/Users/matty_gee/Desktop/Social/SocialCUD
    conda deactivate
    conda activate social_cud

After that, restart any notebook kernels.

DO NOT RENAME THIS FILE.
------------------------
The filename `sitecustomize.py` is special; Python looks for it
explicitly at startup.
"""

import sys
from pathlib import Path

# Absolute path to this repo's /code directory
CODE_DIR = Path(__file__).resolve().parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

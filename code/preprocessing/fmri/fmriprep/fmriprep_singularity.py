#!/usr/bin/env python

# notes
# - templateflow templates are bound directly inside the singularity

import os, glob, shutil, sys, subprocess, logging, datetime, time, json
import pandas as pd
import numpy as np
from copy import deepcopy
import os
import json
import time
import logging
import subprocess

class FmriprepSingularityPipeline(object):
    """
    Prepares batch scripts and runs fmriprep through a Singularity image.
    Designed for use on Minerva at Mount Sinai.

    Methods must be run in order:
      1) create_singularity_batch()
      2) run_singularity_batch()
    """

    def __init__(self,
                 participants,
                 bids_root,
                 output,
                 minerva_options,
                 fmaps=False,
                 freesurfer=False,
                 cifti_output=False,
                 task_id=None,       # None = run all BOLD; str = filter to that task (recommended for debugging)
                 omp_nthreads=1,     # keep ANTs/ITK OMP low to reduce peak memory
                 nthreads=2,         # reduce nipype concurrency to reduce peak memory (was 3)
                 mem_mb=24000,       # match BSUB rusage[mem=...]
                 work_root=None,     # None => default under software_dir
                 use_aroma=False):  
        """
        participants: dict with key 'participant_id' -> list of ids (with or without 'sub-')
        bids_root: BIDS root directory
        output: derivatives output directory
        minerva_options must contain:
          - singularity_location
          - batch_dir
          - project_dir
          - software_dir
        """
        self.subs = participants["participant_id"]
        self.bids_root = bids_root
        self.output = output

        self.freesurfer = freesurfer
        self.fmaps = fmaps
        self.cifti_output = cifti_output

        self.minerva_options = minerva_options
        self.batch_dir = minerva_options["batch_dir"]

        self.task_id = task_id
        self.nthreads = int(nthreads)
        self.omp_nthreads = int(omp_nthreads)
        self.mem_mb = int(mem_mb)
        self.work_root = work_root  # handled in create_singularity_batch()

        self.use_aroma = bool(use_aroma)  # NEW

        if self.cifti_output and not self.freesurfer:
            logging.error("Freesurfer must be on to have cifti-output!")
            raise OSError("Freesurfer must be on to have cifti-output!")

        # Basic validation of required options
        for k in ("singularity_location", "batch_dir", "project_dir", "software_dir"):
            if k not in self.minerva_options:
                raise KeyError(f"minerva_options missing required key: {k}")

    def create_singularity_batch(self):
        """
        Creates subject-specific batch scripts for running fmriprep with Singularity.
        Each subject is run as its own job.
        """
        logging.info("Setting up fmriprep command through Singularity for Minerva")

        # ---------------------------
        # Check singularity image
        # ---------------------------
        img = self.minerva_options["singularity_location"]
        if not os.path.isfile(img):
            logging.error("fmriprep image does not exist in the given location!")
            raise OSError(f"fmriprep image does not exist: {img}")

        # ---------------------------
        # Ensure batch_dir structure exists
        # ---------------------------
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(f"{self.batch_dir}/output", exist_ok=True)
        os.makedirs(f"{self.batch_dir}/jobs", exist_ok=True)

        # ---------------------------
        # TemplateFlow cache checks
        # ---------------------------
        tpl_bind_src = os.path.join(self.minerva_options["software_dir"], "templates")
        if not os.path.isdir(tpl_bind_src):
            raise OSError(
                f"TemplateFlow cache dir not found: {tpl_bind_src}\n"
                "Expected a pre-staged TemplateFlow cache because Minerva is offline."
            )

        # Required templates:
        # - Explicit output space (2009cAsym)
        # - fMRIPrep may still use MNI152NLin6Asym internally for some references/reports/carpet, etc.
        # - OASIS30ANTs is used for brain extraction
        required_tpls = [
            os.path.join(tpl_bind_src, "tpl-MNI152NLin2009cAsym"),
            os.path.join(tpl_bind_src, "tpl-MNI152NLin6Asym"),
            os.path.join(tpl_bind_src, "tpl-OASIS30ANTs"),
        ]
        missing = [p for p in required_tpls if not os.path.isdir(p)]
        if missing:
            raise OSError(
                "Missing required TemplateFlow templates in cache:\n"
                + "\n".join([f"  {p}" for p in missing]) + "\n\n"
                "Minerva is offline; these must be pre-staged under software_dir/templates."
            )

        # FreeSurfer license existence check (cheap + prevents cluster failures)
        fs_lic = os.path.join(self.minerva_options["software_dir"], "license.txt")
        if not os.path.isfile(fs_lic):
            raise OSError(f"FreeSurfer license not found: {fs_lic}")

        # ---------------------------
        # Work + output roots
        # ---------------------------
        work_root = self.work_root or os.path.join(self.minerva_options["software_dir"], "fmriprep_work")
        bids_root = self.bids_root
        out_root = self.output

        os.makedirs(work_root, exist_ok=True)
        os.makedirs(out_root, exist_ok=True)

        # ---------------------------
        # Loop over subjects
        # ---------------------------
        for sub in self.subs:
            sub_label = sub[4:] if str(sub).startswith("sub-") else str(sub)

            batch_script = f"{self.batch_dir}/jobs/sub-{sub_label}.sh"
            with open(batch_script, "w") as f:
                lines = [
                    "#!/bin/bash\n\n",
                    f"#BSUB -J fmriprep_sub-{sub_label}\n",
                    "#BSUB -P acc_guLab\n",
                    "#BSUB -q private\n",
                    f"#BSUB -n {self.nthreads}\n",
                    "#BSUB -W 12:00\n",
                    f"#BSUB -R rusage[mem={self.mem_mb}]\n",
                    f"#BSUB -o {self.batch_dir}/output/sub-{sub_label}.out\n",
                    "#BSUB -L /bin/bash\n\n",
                    "ml singularity/3.6.4\n\n",
                    f"cd {self.minerva_options['project_dir']}\n\n",
                    
                    # Ensure TemplateFlow uses the bind mount and does not attempt updates
                    "export SINGULARITYENV_TEMPLATEFLOW_HOME=/templateflow\n",
                    "export SINGULARITYENV_TEMPLATEFLOW_NO_UPDATE=1\n",
                    "export SINGULARITYENV_TEMPLATEFLOW_NO_PROGRESS=1\n\n",

                    f"export OMP_NUM_THREADS={self.omp_nthreads}\n",
                    f"export SINGULARITYENV_OMP_NUM_THREADS={self.omp_nthreads}\n\n",
                ]
                f.writelines(lines)

                # Per-subject workdir to avoid collisions
                work_dir = f"{work_root}/sub-{sub_label}"
                os.makedirs(work_dir, exist_ok=True)

                # Build fmriprep command
                command = f"""
                    singularity run
                    -B $HOME:/home --home /home
                    -B {self.minerva_options['software_dir']}:/software
                    -B {tpl_bind_src}:/templateflow
                    -B {bids_root}:{bids_root}
                    -B {out_root}:{out_root}
                    -B {work_root}:{work_root}
                    --cleanenv
                    {img}
                    {bids_root}
                    {out_root}
                    participant
                    --output-spaces MNI152NLin2009cAsym:res-2
                    --participant-label {sub_label}
                    -w {work_dir}
                    --nthreads {self.nthreads}
                    --omp-nthreads {self.omp_nthreads}
                    --mem-mb {self.mem_mb}
                    --low-mem
                    --skip_bids_validation
                    --notrack
                    --fs-license-file {fs_lic}
                """
                command = " ".join(command.split())

                # Optional: filter to a single task (recommended to match your older working setup)
                if self.task_id:
                    command = " ".join([command, f"--task-id {self.task_id}"])

                if not self.fmaps:
                    command = " ".join([command, "--ignore fieldmaps"])
                if not self.freesurfer:
                    command = " ".join([command, "--fs-no-reconall"])
                if self.cifti_output:
                    command = " ".join([command, "--cifti-output"])

                # NEW: AROMA optional
                if self.use_aroma:
                    command = " ".join([command, "--use-aroma"])

                f.write(command + "\n")

        # ---------------------------
        # Save parameters used
        # ---------------------------
        self.minerva_options["subs"] = self.subs
        self.minerva_options["bids_root"] = self.bids_root
        self.minerva_options["output"] = self.output
        self.minerva_options["freesurfer"] = self.freesurfer
        self.minerva_options["task_id"] = self.task_id
        self.minerva_options["omp_nthreads"] = self.omp_nthreads
        self.minerva_options["nthreads"] = self.nthreads
        self.minerva_options["mem_mb"] = self.mem_mb
        self.minerva_options["work_root"] = self.work_root
        self.minerva_options["templateflow_cache"] = tpl_bind_src
        self.minerva_options["use_aroma"] = self.use_aroma  # NEW

        with open(f"{self.batch_dir}/minerva_options.json", "w") as f:
            json.dump(self.minerva_options, f, indent=2)

    def run_singularity_batch(self, overwrite=False):
        """
        Submits generated subject batch scripts to LSF via bsub.
        overwrite=False => skip subjects that already have output/sub-XX/
        """
        logging.info("Submitting singularity batch scripts to the queue")
        counter = 1

        for sub in self.subs:
            sub_label = sub[4:] if str(sub).startswith("sub-") else str(sub)

            if os.path.isdir(f"{self.output}/sub-{sub_label}/"):
                logging.warning(f"sub-{sub_label} preprocessing already completed!")
                if not overwrite:
                    logging.info(f"Skipping sub-{sub_label}")
                    continue
                logging.warning(f"Re-preprocessing sub-{sub_label}, and overwriting results!")

            logging.info(f"Submitting Job {counter} of {len(self.subs)}")
            subprocess.run(f"bsub < {self.batch_dir}/jobs/sub-{sub_label}.sh", shell=True)
            counter += 1
            time.sleep(20)

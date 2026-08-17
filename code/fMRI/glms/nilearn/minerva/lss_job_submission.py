import os
import time
import logging
import subprocess
from pathlib import Path

def main():

    # ----------------- set up -----------------

    base_dir = '/sc/arion/projects/OlfMem/mgs'
    code_dir = f'{base_dir}/code'
    batch_dir = code_dir

    for dir_ in ['jobs', 'out']:
        os.makedirs(f'{batch_dir}/{dir_}', exist_ok=True)

    projects = ['acc_guLab', 'acc_OlfMem']
    queues   = ['private', 'premium']
    submit_to = 0
    overwrite = False
    n_jobs = 1
    glm_n_jobs = 1
    n_cpus = int(n_jobs) * int(glm_n_jobs)

    # ----------------- paths -----------------

    behav_dir   = f'{base_dir}/data/preprocessed/behavior'
    preprc_dir  = f'{base_dir}/data/preprocessed/fmriprep/derivatives-fmap/fmriprep'
    glm_dir     = f'{base_dir}/analyses/lss_decision_fmriprep/glms'
    timing_file = f'{code_dir}/timing.xlsx'

    os.makedirs(glm_dir, exist_ok=True)

    # ----------------- find subjects -----------------

    sub_ids = [
        d for d in os.listdir(preprc_dir)
        if d.startswith('sub-') and os.path.isdir(os.path.join(preprc_dir, d))
    ]
    sub_ids.sort()

    print(f'Found {len(sub_ids)} subjects in {preprc_dir}')

    # ----------------- submit jobs -----------------

    for sub_id in sub_ids:

        sub_out_dir     = Path(glm_dir) / sub_id
        required_outputs = [
            sub_out_dir / 'decision_trials_beta.nii',
            sub_out_dir / 'narrative_trials_beta.nii',
            sub_out_dir / 'all_trials_beta.nii',
        ]

        # skip only if all final beta outputs already exist
        if all(p.exists() for p in required_outputs) and not overwrite:
            print(f'Skipping {sub_id}: all final beta outputs already exist')
            continue

        print(f'Submitting LSS job for {sub_id}')

        batch_sh = f'{batch_dir}/jobs/{sub_id}_lss_decision.sh'
        with open(batch_sh, 'w') as f:
            cookies = [
                '#!/bin/bash\n\n',
                f'#BSUB -J {sub_id}_lss\n',
                f'#BSUB -P {projects[submit_to]}\n',
                f'#BSUB -q {queues[submit_to]}\n',
                f'#BSUB -n {n_cpus}\n',
                f'#BSUB -W 06:00\n',
                f'#BSUB -R rusage[mem=64000]\n',
                f'#BSUB -o {batch_dir}/out/{sub_id}_lss.out\n',
                f'#BSUB -L /bin/bash\n\n',
                'ml python\n\n',
                'export OMP_NUM_THREADS=1\n',
                'export OPENBLAS_NUM_THREADS=1\n',
                'export MKL_NUM_THREADS=1\n',
                'export VECLIB_MAXIMUM_THREADS=1\n',
                'export NUMEXPR_NUM_THREADS=1\n',
                'export BLIS_NUM_THREADS=1\n\n',
                f'cd {code_dir}\n\n',
            ]
            f.writelines(cookies)
            py_cmd = (
                "python -c 'from lss_utils import run_lss_subject; run_lss_subject("
                f"sub_id=\"{sub_id}\", "
                f"preprc_dir=r\"{preprc_dir}\", "
                f"behav_dir=r\"{behav_dir}\", "
                f"timing_file=r\"{timing_file}\", "
                f"glm_dir=r\"{glm_dir}\", "
                "tr=1.0, "
                "slice_time_ref=0.5, "
                "high_pass=1/128, "
                f"n_jobs={n_jobs}, "
                f"glm_n_jobs={glm_n_jobs}"
                ")'\n"
            )
            f.write(py_cmd)

        logging.info(f'Submitting Job: {sub_id}')
        subprocess.run(f'bsub < {batch_sh}', shell=True)
        time.sleep(2)

if __name__ == '__main__':
    main()

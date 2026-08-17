#!/usr/bin/env python

# NOTE: have to run [ $ ml python ] in terminal first
import json, os, glob, sys, time, subprocess, logging
import pandas as pd

def main():

    #----------------- set up

    base_dir = '/sc/arion/projects/OlfMem/mgs'

    # for job files
    batch_dir = f'{base_dir}/code' # /submit_jobs/searchlights'
    for dir_ in ['jobs', 'out']:
        if not os.path.exists(f'{batch_dir}/{dir_}'):
            os.makedirs(f'{batch_dir}/{dir_}')

    # diff options for job submission
    projects  = ['acc_guLab', 'acc_OlfMem']
    queues    = ['private', 'premium']
    submit_to = 1

    #----------------- find subjects

    # find included subject directories
    glm_dir  = f'{base_dir}/analyses/lss_decision_fmriprep'
    sub_dirs = glob.glob(f'{glm_dir}/glms/sub-*')
    sub_dirs.sort()

    #----------------- submit jobs

    for sl_model in ['dimension_regression']:

        print(f'Running {len(sub_dirs)} {sl_model} searchlight jobs...')
        for sub_dir in sub_dirs:
            
            # find fnames based on sample
            sub_id     = os.path.basename(sub_dir)  
            func_fname = f'{sub_dir}/decision_trials_beta.nii.gz'
            mask_fname = f'{sub_dir}/mask.nii.gz'

            print(f'Running {sub_id} {sl_model} searchlight...')
            print(f' - Functional file: {func_fname}')
            print(f' - Mask file: {mask_fname}')

            # create the job
            batch_sh = f'{batch_dir}/jobs/{sub_id}_{sl_model}_sl.sh'
            with open(batch_sh, 'w') as f:
                cookies = [ f'#!/bin/bash\n\n',
                            f'#BSUB -J {sub_id}_sl\n',
                            f'#BSUB -P {projects[submit_to]}\n',
                            f'#BSUB -q {queues[submit_to]}\n',
                            f'#BSUB -n 3\n',
                            f'#BSUB -W 03:00\n',
                            f'#BSUB -R rusage[mem=8000]\n',
                            f'#BSUB -o {batch_dir}/out/{sub_id}_{sl_model}_sl.out\n',
                            f'#BSUB -L /bin/bash\n\n',
                            f'ml python\n\n', 
                            f'ml brainiak/0.12\n\n',
                            f'cd {base_dir}/code/\n\n']
                f.writelines(cookies)
                f.write(
                    f"python -c 'from searchlight_utils import run_{sl_model}_searchlight; "
                    f"run_{sl_model}_searchlight(\"{func_fname}\", mask_fname=\"{mask_fname}\", glm_name=\"lss_decision_fmriprep\")'"
                )

            # submit the job
            logging.info(f'Submitting Job: {sub_id}')
            subprocess.run(f'bsub < {batch_sh}', shell=True)
            time.sleep(20)

if __name__ == '__main__':
    main()
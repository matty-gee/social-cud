import pandas as pd
import numpy as np
import os, glob, shutil, sys, subprocess, logging, datetime, time, json
from copy import deepcopy
import fmriprep_singularity as fs

project_dir  = '/sc/arion/projects/OlfMem/mgs'
bids_root    = f'{project_dir}/data/BIDS'
code_dir     = f'{project_dir}/code/fmriprep'
batch_dir    = f'{code_dir}/batch_dir'
software_dir = f'{code_dir}/fmriprep_software'
singularity  = f'{software_dir}/fmriprep-20.2.0.simg'
work_dir     = f'{software_dir}/fmriprep_work'

# use fieldmaps 
fmaps = True
if fmaps:
    output_dir = f'{project_dir}/data/preprocessed/fmriprep/derivatives-fmap/'
else:
    output_dir = f'{project_dir}/data/preprocessed/fmriprep/derivatives/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# define participants
pd_participants = pd.read_csv(f'{code_dir}/participants.tsv', sep='\t')
if 'id' in pd_participants.columns:
    subj_list = list(pd_participants['id'])
elif 'participant_id' in pd_participants.columns:
    subj_list = list(pd_participants['participant_id'])
else:
    raise ValueError(f"participants.tsv missing expected column. Found: {list(pd_participants.columns)}")
participants = {'participant_id': subj_list}

# define minerva options
minerva_options = {
    'singularity_location': singularity,
    'code_dir': code_dir,
    'software_dir': software_dir,
    'batch_dir': batch_dir,
    'project_dir': project_dir
}

# run it
fp_singularity = fs.FmriprepSingularityPipeline(
    participants,
    bids_root,
    output_dir,
    minerva_options,
    fmaps=fmaps,
    use_aroma=False,
    nthreads=2,            # concurrency
    omp_nthreads=1,        # keep OMP low
    mem_mb=64000,        
    work_root=work_dir,
)
fp_singularity.create_singularity_batch()
fp_singularity.run_singularity_batch() 